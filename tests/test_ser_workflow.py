import torch
import pytest
from pathlib import Path
from human_voice_ai.utils.config import Config,AudioConfig,SERConfig
from human_voice_ai.emotion.ser_model import SERModel,create_ser_model
from human_voice_ai.ser_smoke import run


def tiny():
    torch.set_num_threads(1)
    return Config(device="cpu",audio=AudioConfig(n_mels=16),ser=SERConfig(input_dim=16,hidden_dim=8,num_layers=1,num_classes=6))


def test_packaged_yaml_nested_config_and_constructor(tmp_path):
    cfg=tiny();path=tmp_path/'config.yaml';cfg.to_yaml(path)
    restored=Config.from_yaml(path);model=SERModel(restored)
    assert model.config.audio.n_mels==16
    assert all(int(m.num_batches_tracked)==0 for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    with pytest.raises(ValueError):SERModel(Config(audio=AudioConfig(n_mels=16)))


def test_real_acoustic_train_checkpoint_policy(tmp_path):
    report=run(tmp_path,Path(__file__).parent/'fixtures/test_policy_config.yaml')
    assert report['parameter_l1_change']>0
    assert report['optimizer_updates']==3
    assert report['checkpoint_predictions_match']
    assert report['policy_action'] is not None


def test_extended_features_forward_backward_and_reload(tmp_path):
    cfg=tiny();model=create_ser_model(cfg,feature_dim=3)
    x=torch.randn(2,1,16,24);f=torch.randn(2,3)
    model(x,f)['logits'].sum().backward()
    assert model.feature_processor[0].weight.grad is not None
    with pytest.raises(ValueError):model(x)
    model.save_checkpoint(tmp_path/'extended.pt')
    restored,_=type(model).load_checkpoint(tmp_path/'extended.pt',feature_dim=3,device='cpu')
    model.eval();restored.eval()
    torch.testing.assert_close(model(x,f)['probs'],restored(x,f)['probs'],rtol=0,atol=0)


def test_input_contract_and_attention_toggle():
    cfg=tiny();cfg.ser.use_attention=False;model=SERModel(cfg)
    x=torch.randn(2,1,16,24);out=model(x)
    torch.testing.assert_close(out['attention_weights'],torch.full_like(out['attention_weights'],1/out['attention_weights'].size(1)))
    with pytest.raises(ValueError):model(torch.full_like(x,float('nan')))
    with pytest.raises(ValueError):model(torch.randn(2,1,16,4))
