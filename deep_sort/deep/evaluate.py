import torch

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
res = scores.topk(5, dim=1)[1][:,0]
top1correct = gl[res].eq(ql).sum().item()

print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


# ckpt.t7 Acc top1:0.985
# 512     Acc top1:0.986
# 256     Acc top1:0.985
# 128     Acc top1:0.980
