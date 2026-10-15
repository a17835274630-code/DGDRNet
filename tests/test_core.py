"""Run with: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dgdrnet import DGDRNet,ModelConfig,RestorationLoss
from dgdrnet.model import DFEM,DSMEE,PriorConv,gaussian_high_pass,laplacian_kernel
from dgdrnet.data import PairedDataset,bounding_box_mask
from dgdrnet.degradation import Parameters,motion_blur_kernel,sample_parameters,per_image_rng,degrade_bgr
from dgdrnet.metrics import psnr,ssim


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(3407)

    def test_model_shapes_and_backward(self):
        for options in ({},{'use_dsmee':False,'use_dfem':False},{'use_dfem':False},
                        {'edge_guidance':False},{'branches':('small',)},{'branches':('medium','large')}):
            model=DGDRNet(ModelConfig(width=4,gate_hidden=4,**options))
            image=torch.rand(1,3,16,24)
            pred=model(image)
            self.assertEqual(pred.shape,image.shape)
            pred.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):DGDRNet(ModelConfig(use_dsmee=False,use_dfem=True))

    def test_dsmee_absolute_responses_and_kernel_priors(self):
        block=DSMEE(2)
        _,aux=block(torch.randn(1,2,8,8))
        self.assertTrue(all((r>=0).all() for r in aux['edges']))
        self.assertTrue(torch.allclose(aux['weights'].sum(1),torch.ones(1,8,8)))
        kernel=gaussian_high_pass()
        axis=torch.arange(7)-3
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        gaussian=torch.exp(-(x*x+y*y)/(2*1.5**2));gaussian/=gaussian.sum()
        impulse=torch.zeros(7,7);impulse[3,3]=1
        self.assertTrue(torch.allclose(kernel,impulse-gaussian))

    def test_dfem_formula_and_constant_mask(self):
        x=torch.randn(2,3,8,10)
        mask=torch.rand(2,1,8,10)
        module=DFEM(3,4)
        result,aux=module(x,mask)
        self.assertTrue(torch.allclose(aux['W_MS'],aux['S']*aux['G']))
        self.assertTrue(torch.allclose(aux['enhanced_magnitude'],aux['magnitude']*(1+aux['W_MS'])))
        spec=torch.fft.rfft2(x,norm='ortho')
        expected=x+module.project(torch.fft.irfft2(torch.polar(aux['enhanced_magnitude'],torch.angle(spec)),s=x.shape[-2:],norm='ortho'))
        self.assertTrue(torch.allclose(result,expected))
        _,constant=module(x,torch.ones_like(mask))
        self.assertEqual(constant['S'].abs().max().item(),0)
        self.assertTrue(torch.equal(constant['enhanced_magnitude'],constant['magnitude']))
        unguided=DFEM(3,4,guided=False)
        _,a=unguided(x,mask)
        _,b=unguided(x,torch.ones_like(mask))
        self.assertTrue(torch.equal(a['W_MS'],a['G']))
        self.assertTrue(torch.equal(a['W_MS'],b['W_MS']))
        self.assertGreater(a['W_MS'].min().item(),0)

    def test_losses_against_direct_formula(self):
        model=DGDRNet(ModelConfig(width=4,gate_hidden=4))
        criterion=RestorationLoss(edge_weight=0,spectral_weight=0,kernel_weight=0)
        criterion.set_reference(1.0)
        prediction=torch.full((2,3,8,8),0.2,requires_grad=True)
        target=torch.zeros_like(prediction)
        mask=torch.zeros(2,1,8,8);mask[0,:,:4,:4]=1;mask[1]=1
        loss,terms=criterion(prediction,target,mask,model)
        error=torch.sqrt(prediction.square()+1e-6)
        ema=0.95+0.05*error.mean().detach()
        gamma=1-ema
        alpha=torch.tensor([gamma*np.log(4),0.]).reshape(2,1,1,1)
        expected=((1+alpha*mask)*error).mean()
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward()
        self.assertIsNone(criterion.ema.grad)
        empty=torch.zeros_like(mask)
        value,_=criterion(prediction.detach(),target,empty,model,update_state=False)
        self.assertTrue(torch.allclose(value,error.detach().mean()))
        old=criterion.ema.clone();criterion.eval()
        criterion(prediction.detach(),target,mask,model)
        self.assertTrue(torch.equal(old,criterion.ema))
        criterion.train()
        direct=(torch.fft.fft2(prediction.detach(),norm='ortho')-torch.fft.fft2(target,norm='ortho')).abs().mean()
        self.assertTrue(torch.allclose(criterion.spectral_loss(prediction.detach(),target),direct))

    def test_phase_sensitive_spectral_loss(self):
        a=torch.zeros(1,3,8,8);a[:,:,1,1]=1
        b=a.roll(1,dims=-1)
        criterion=RestorationLoss(weighted_pixel=False)
        self.assertTrue(torch.allclose(torch.fft.fft2(a).abs(),torch.fft.fft2(b).abs(),atol=1e-6))
        self.assertGreater(criterion.spectral_loss(a,b).item(),0)

    def test_kernel_loss_is_frobenius_sum(self):
        model=DGDRNet(ModelConfig(width=4,gate_hidden=4))
        self.assertEqual(model.kernel_consistency_loss().item(),0)
        prior=next(m for m in model.modules() if isinstance(m,PriorConv))
        with torch.no_grad():prior.weight.add_(0.1)
        self.assertAlmostEqual(model.kernel_consistency_loss().item(),prior.weight.numel()*0.01,places=5)
        criterion=RestorationLoss(weighted_pixel=False)
        self.assertTrue(torch.equal(criterion.laplacian[0,0],laplacian_kernel()))

    def test_data_pairing_and_binary_union(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('sharp','blur','labels'):(root/name).mkdir()
            pixels=np.full((20,30,3),128,np.uint8)
            for name in ('sharp','blur'):Image.fromarray(pixels).save(root/name/'sample.png')
            (root/'labels'/'sample.txt').write_text('0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.5 0.5\n',encoding='utf-8')
            mask=bounding_box_mask(root/'labels'/'sample.txt',16,16)
            self.assertEqual(mask.sum().item(),64)
            data=PairedDataset(root/'blur',root/'sharp',root/'labels',16)
            sample=data[0]
            self.assertEqual(sample['blur'].shape,(3,16,16))
            self.assertTrue(torch.equal(sample['mask'],mask))
            no_boxes=PairedDataset(root/'blur',root/'sharp',size=16)[0]['mask']
            self.assertEqual(no_boxes.sum().item(),0)
            Image.fromarray(pixels).save(root/'sharp'/'unpaired.png')
            with self.assertRaises(ValueError):PairedDataset(root/'blur',root/'sharp',size=16)

    def test_degradation_deterministic_and_ranges(self):
        for length in (7,8,30,65):
            kernel=motion_blur_kernel(length,33)
            self.assertAlmostEqual(float(kernel.sum()),1,places=6)
            self.assertEqual(kernel.shape[0],length if length%2 else length+1)
        rng=per_image_rng(3407,'sample')
        for _ in range(200):
            p=sample_parameters(rng)
            self.assertTrue(7<=p.blur_length<=65 and 0<=p.blur_angle<180 and 1<=p.noise_sigma<=54 and 50<=p.jpeg_quality<=95)
        pixels=np.full((32,40,3),128,np.uint8)
        p=Parameters(8,45,10,80)
        a,encoded=degrade_bgr(pixels,p,per_image_rng(3407,'sample'))
        b,other=degrade_bgr(pixels,p,per_image_rng(3407,'sample'))
        self.assertTrue(np.array_equal(a,b) and np.array_equal(encoded,other))
        self.assertEqual(a.shape,pixels.shape)

    def test_metrics_identity(self):
        a=torch.rand(2,3,16,16)
        self.assertTrue(torch.allclose(ssim(a,a),torch.ones(2),atol=1e-6))
        self.assertTrue(torch.allclose(psnr(a,a),torch.full((2,),120.)))


if __name__=='__main__':unittest.main()
