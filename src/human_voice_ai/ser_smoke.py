"""Actual CPU acoustic/SER training fixture; arbitrary labels, no accuracy claim."""
import argparse
import json
from pathlib import Path
import torch
from .audio.feature_extractor import FeatureExtractor
from .emotion.ser_model import SERModel
from .utils.config import Config, AudioConfig, SERConfig
from .policy.manager import PolicyManager


def run(output="artifacts/ser-smoke", policy_config=None):
    torch.set_num_threads(1);torch.manual_seed(7)
    config=Config(device="cpu",audio=AudioConfig(sample_rate=16000,n_fft=128,hop_length=64,
                 win_length=128,n_mels=16),ser=SERConfig(input_dim=16,hidden_dim=8,num_layers=1,num_classes=6))
    extractor=FeatureExtractor(sample_rate=16000,n_fft=128,hop_length=64,win_length=128,n_mels=16)
    time=torch.arange(4096)/16000
    waveforms=[.1*torch.sin(2*torch.pi*frequency*time)+.002*torch.randn_like(time) for frequency in [220,440]]
    full_features=[extractor.extract_from_waveform(wave,16000) for wave in waveforms]
    # CNN SER consumes mel bins, while pitch/energy remain explicit extractor extras.
    batch=torch.stack([features[:,:16].T for features in full_features]).unsqueeze(1)
    if not torch.isfinite(batch).all():raise RuntimeError("Non-finite acoustic features")
    model=SERModel(config);initial={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);losses=[]
    for _ in range(3):
        model.train();optimizer.zero_grad();out=model(batch)
        loss=torch.nn.functional.cross_entropy(out["logits"],torch.tensor([0,1]))
        if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError("Non-finite gradient")
        optimizer.step();losses.append(float(loss.detach()))
    delta=sum(float((p-initial[n]).abs().sum().detach()) for n,p in model.named_parameters())
    if delta<=0:raise RuntimeError("Optimizer did not change model")
    destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    model.save_checkpoint(destination/"ser.pt",optimizer=optimizer,epoch=3,fixture_scope="synthetic tones/arbitrary labels")
    restored,checkpoint=SERModel.load_checkpoint(destination/"ser.pt",device="cpu")
    original_probs,_=model.predict_emotion(batch);probs,preds=restored.predict_emotion(batch)
    torch.testing.assert_close(probs,original_probs,rtol=0,atol=0)
    if not torch.allclose(probs.sum(-1),torch.ones(2)):raise RuntimeError("Invalid probabilities")
    action=None
    if policy_config:
        manager=PolicyManager(config_path=str(policy_config));manager.policy.epsilon=0
        action=manager.get_action(probs[0].numpy()).id
    report={"complete":True,"scope":"Synthetic tone acoustic features -> actual SER3optimizer updates -> strict checkpoint restore -> inference",
            "not_emotion_accuracy":True,"initial_state":"random initialization","trained_fixture_state":True,
            "labels":"arbitrary0/1labels; no semantic emotion validation","sample_rate":16000,
            "feature_shapes":[list(f.shape) for f in full_features],"mel_batch_shape":list(batch.shape),
            "parameters":model.count_parameters(),"optimizer_updates":3,"losses":losses,
            "parameter_l1_change":delta,"checkpoint_predictions_match":True,"probabilities":probs.tolist(),
            "policy_action":action,"live_tts":False,"live_microphone":False}
    (destination/"receipt.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default="artifacts/ser-smoke")
    p.add_argument("--policy-config");a=p.parse_args();run(a.output,a.policy_config)
if __name__=="__main__":main()
