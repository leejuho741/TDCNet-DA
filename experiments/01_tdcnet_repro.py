# ============================================================
# B4: TDCNet 재현 맞추기 — fp32 + 공식 TransCG 증강으로 다시 학습 (L4 약 10~12시간, 런타임 유지)
#   배경: B2 TDCNet은 12.49로 예전 재현(11.91 ± 0.16)·논문(12)보다 0.6 나쁨. B2-b에서 평가 쪽 원인(입력 1.5 m 자르기)은 아니었음
#         → 학습 쪽 차이(bf16, 우리 로더의 색 증강)를 예전 재현과 같게 되돌림. 나머지(레시피·평가 코드)는 B2와 같음
#   평가: B2와 같은 평가 + 최종안 후보(DA-V2-Small·Base head x 부드러운 합성 2→6, 4→12 px)
#   결과: MyDrive/tdc_finetune/b2_tdcnet/ (log_repro_fp32.txt, eval_repro_fp32*.json, repro_fp32/final.tar)
#   중간에 끊기면 다시 실행: 마지막 epoch부터 이어감
# ============================================================
import os, sys, glob, json, shutil, subprocess, importlib

REPO     = '/content/ReMake'
CK_DIR   = '/content/drive/MyDrive/tdc_finetune/remake_official'
CKPT     = f'{CK_DIR}/remake.tar'
ORIG_DIR = ''                                     # 원본 TransCG 폴더(장면 폴더들이 있는 곳). 없으면 비워 둠
CACHE_ARCHIVE, CACHE_ROOT = '/content/drive/MyDrive/transcg_320.zip', '/content/tcg320_full'


def run(cmd, cwd=None):
    """명령을 실행하면서 출력을 셀에 바로 찍음 (지난번 탐색 스크립트는 출력이 안 보였음)"""
    print(f'$ {cmd}')
    # PyTorch 2.6부터 torch.load 기본값이 weights_only=True라 공식 체크포인트(NumPy 값 포함)를 거부함 →
    # 공식 저장소에서 받은 파일이므로 예전 동작(weights_only=False)으로 돌리는 환경 변수를 켬
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end='')
    return p.wait()


if not os.path.ismount('/content/drive'):
    if os.path.isdir('/content/drive'): shutil.rmtree('/content/drive')
    from google.colab import drive; drive.mount('/content/drive')
import torch; print('GPU', torch.cuda.get_device_name(), '| torch', torch.__version__)

# ---------- 1) 저장소 + 패키지 ----------
if not os.path.isdir(REPO): run(f'git clone -q https://github.com/ChengYaofeng/ReMake.git {REPO}')
run(f'git -C {REPO} log -1 --format="ReMake 커밋 %h %cd"')
for mod, pipname in [('open3d', 'open3d'), ('yaml', 'pyyaml'), ('tqdm', 'tqdm'), ('einops', 'einops'), ('timm', 'timm'),
                     ('tensorboardX', 'tensorboardX'), ('termcolor', 'termcolor'), ('OpenEXR', 'OpenEXR'), ('Imath', 'OpenEXR'),
                     # 테스트에는 안 쓰지만 공식 코드가 import 단계에서 불러오는 것들 (시각화·SAM·RealSense)
                     ('pytorch_grad_cam', 'grad-cam'), ('segment_anything', 'git+https://github.com/facebookresearch/segment-anything.git'),
                     ('pyrealsense2', 'pyrealsense2'), ('monai', 'monai')]:
    try:
        importlib.import_module(mod)
    except Exception:
        run(f'pip -q install {pipname}')
run('python -c "import numpy; print(\'numpy\', numpy.__version__)"')
# Colab에 깔린 Hugging Face `datasets` 패키지가 저장소의 datasets/ 폴더를 가림 → 저장소 폴더들을 정식 패키지로
for d in ['datasets', 'utils', 'run_utils', 'run_tools', 'models', 'relat_depth_models']:
    f = f'{REPO}/{d}/__init__.py'
    if os.path.isdir(f'{REPO}/{d}') and not os.path.exists(f): open(f, 'w').close(); print('추가:', f)

# ---------- 1.5) ReMake 공식 체크포인트 (없으면 Google Drive 공개 링크에서 받음) ----------
if not os.path.exists(CKPT):
    os.makedirs(CK_DIR, exist_ok=True); run('pip -q install gdown')
    run(f'cd "{CK_DIR}" && gdown --fuzzy "https://drive.google.com/file/d/1AF9sWyhEoNHnAlJ_y1AXsoxgrXq44333/view?usp=drive_link"')
    found = [f for f in glob.glob(f'{CK_DIR}/**/*', recursive=True) if f.endswith(('.tar', '.pth', '.pt', '.ckpt'))]
    if not found:
        for z in glob.glob(f'{CK_DIR}/*.zip'): run(f'cd "{CK_DIR}" && unzip -o -q "{z}"')
        found = [f for f in glob.glob(f'{CK_DIR}/**/*', recursive=True) if f.endswith(('.tar', '.pth', '.pt', '.ckpt'))]
    assert found, f'체크포인트를 못 받음: {CK_DIR} 내용을 확인해 주세요'
    CKPT = found[0]
print('ReMake 체크포인트:', CKPT, f'({os.path.getsize(CKPT) / 1e6:.1f} MB)')

# ---------- 2) Depth Anything V2 Small 가중치 ----------
os.makedirs(f'{REPO}/checkpoints', exist_ok=True)
DA_W = f'{REPO}/checkpoints/depth_anything_v2_vits.pth'
if not os.path.exists(DA_W):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('depth-anything/Depth-Anything-V2-Small', 'depth_anything_v2_vits.pth', local_dir=f'{REPO}/checkpoints')
    print('Depth Anything 가중치', p)

# ---------- 3) 데이터 연결 (datasets/transcg/transcg) ----------
if ORIG_DIR and os.path.exists(f'{ORIG_DIR}/metadata.json'):
    DATA, NOTE = ORIG_DIR, '원본 TransCG (논문과 같은 조건)'
else:
    if len(glob.glob(f'{CACHE_ROOT}/**/scene*/metadata.json', recursive=True)) < 100:
        os.makedirs(CACHE_ROOT, exist_ok=True); print('320 캐시 복사 중 (약 7분)...')
        run(f'cp "{CACHE_ARCHIVE}" /content/_cache.zip && unzip -q -o /content/_cache.zip -d {CACHE_ROOT} && rm /content/_cache.zip')
    DATA = os.path.dirname(sorted(glob.glob(f'{CACHE_ROOT}/**/scene1/metadata.json', recursive=True), key=len)[0].rsplit('/', 1)[0])
    NOTE = '우리 320x240 캐시'
    if not os.path.exists(f'{DATA}/metadata.json'):              # 공식 로더가 요구하는 최상위 metadata.json 생성
        meta = {'train': [], 'test': [], 'train_samples': 0, 'test_samples': 0}
        for d in glob.glob(f'{DATA}/scene*'):
            sm = json.load(open(f'{d}/metadata.json')); sid = int(os.path.basename(d)[5:]); meta[sm['split']].append(sid)
            meta[f"{sm['split']}_samples"] += len(sm['D435_valid_perspective_list']) + len(sm['L515_valid_perspective_list'])
        meta['train'].sort(); meta['test'].sort(); meta.update(total_scenes=len(meta['train']) + len(meta['test']), perspective_num=240)
        json.dump(meta, open(f'{DATA}/metadata.json', 'w'))
    if not os.path.exists(f'{DATA}/camera_intrinsics/2-camIntrinsics-L515.npy'):
        os.makedirs(f'{DATA}/camera_intrinsics', exist_ok=True)
        for fn in ['1-camIntrinsics-D435.npy', '2-camIntrinsics-L515.npy']:
            src = sorted(glob.glob(f'{CACHE_ROOT}/**/{fn}', recursive=True) +
                         glob.glob(f'/content/drive/MyDrive/Colab Notebooks/FDCT/transcg-info/**/{fn}', recursive=True), key=len)[0]
            shutil.copy(src, f'{DATA}/camera_intrinsics/')
print('데이터:', DATA, '|', NOTE)
link = f'{REPO}/datasets/transcg/transcg'
os.makedirs(os.path.dirname(link), exist_ok=True)
if os.path.islink(link) or os.path.exists(link):
    if os.path.islink(link): os.unlink(link)
os.symlink(DATA, link)


# ==================== B2 본체 ====================
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import time, math, random, inspect, traceback
import numpy as np, cv2, yaml, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
if not hasattr(np, 'bool'): np.bool = bool

# ---------- 설정 ----------
VARIANT = 'plain'                 # 모델은 TDCNet 그대로
RUN = 'repro_fp32'                # 저장 이름 (B2 plain과 구분)
FP32 = True                       # B2는 bf16 → 이번엔 fp32 (TF32 허용, 예전 재현과 같음)
OFFICIAL_AUG = True               # B2는 우리 로더(밝기·대비 등) → 이번엔 공식 TransCG 로더(색상 변환 + 잡음/모션블러, 뒤집기)
EPOCHS, BS, LR, MILESTONES = 40, 8, 1e-3, [35]   # TDCNet 공식 레시피 (scratch, AdamW, epoch 35에서 x0.1)
BETA = 0.1                        # 법선 loss 가중치 (공식)
HEAD_EPOCHS, HEAD_BS, HEAD_LR, PW = 10, 16, 1e-3, 10.0   # head를 불러올 때 확인용 (B1과 같음)
H, W = 240, 320
DA_SIZE = (252, 336)
AUTO_SHUTDOWN = False             # 낮 작업: 런타임 유지
TK = '/content/drive/MyDrive/tdc_finetune'
OUT = f'{TK}/b2_tdcnet'
HEAD_FULL = f'{TK}/a1a2_mask/da_mask_head.tar'
B1_DIR = f'{TK}/b1_robust'        # ReMake 결과 (요약 표에 같이)
os.makedirs(OUT, exist_ok=True); dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
T_START = time.time()


class _Tee:
    def __init__(self, s, f): self.s, self.f = s, f
    def write(self, x): self.s.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.s.flush(); self.f.flush()
    def __getattr__(self, k): return getattr(self.s, k)


if not isinstance(sys.stdout, _Tee): sys.stdout = _Tee(sys.stdout, open(f'{OUT}/log_{RUN}.txt', 'a'))
el = lambda: f'{(time.time() - T_START) / 3600:.2f}h'
print(f'\n######## B4 재현 맞추기 ({RUN}) 시작 {time.strftime("%m-%d %H:%M")} | {torch.cuda.get_device_name()} | AMP {AMP} | 데이터: {DATA}')

# ---------- TDCNet (ReMake 저장소에 들어 있는 TDCNet 코드 = 공식과 같은 구조, BN eps 64/128 특이점 포함) ----------
os.chdir(REPO); sys.path.insert(0, REPO)
for k in [k for k in list(sys.modules) if k.split('.')[0] in ('datasets', 'utils', 'models', 'run_utils', 'run_tools')]: sys.modules.pop(k)
from models.tdcnet import TDCNet
from utils.tools import get_surface_normal_from_xyz


class Net(nn.Module):
    """forward(img, rel, depth). plain은 rel을 무시. rel은 상대 depth를 이미지마다 표준화해 각 단계에 1x1/3x3 conv(마지막 0 초기화)로 더함"""
    def __init__(self, variant):
        super().__init__(); self.variant = variant; self.base = TDCNet(lambda_val=1, res=True)
        if variant == 'rel':
            self.base.eval()
            with torch.no_grad(): chs = [e.shape[1] for e in self.base.encoder(torch.zeros(1, 4, H, W))]
            self.base.train(); self.proj = nn.ModuleList()
            for c in chs:
                m = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 1)); nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
                self.proj.append(m)

    def forward(self, img, rel, depth):
        if self.variant == 'plain': return self.base(img, depth)
        n, h, w = depth.shape; d = depth.view(n, 1, h, w)
        enc = self.base.encoder(torch.cat((img, d), 1)); res = self.base.resnet(d)
        r = (rel - rel.mean((2, 3), keepdim=True)) / (rel.std((2, 3), keepdim=True) + 1e-6)
        for i in range(4): enc[i] = enc[i] + res[i] + self.proj[i](F.interpolate(r, size=enc[i].shape[-2:], mode='bilinear', align_corners=False))
        return self.base.final(self.base.decoder(enc))


with torch.no_grad():
    _m = Net(VARIANT).to(dev).eval(); _o = _m(torch.rand(2, 3, H, W, device=dev), torch.rand(2, 1, H, W, device=dev), torch.rand(2, H, W, device=dev))
    print(f'TDCNet ({VARIANT}) 출력 {tuple(_o.shape)}, 파라미터 {sum(p.numel() for p in _m.parameters()) / 1e6:.2f}M'); del _m, _o

# ---------- Depth Anything V2 Small: 한 번 돌려서 상대 depth + head 특징 ----------
from transformers import AutoModelForDepthEstimation
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); _STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
da = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').to(dev).eval()
for p in da.parameters(): p.requires_grad = False
_FM = {}
da.backbone.register_forward_hook(lambda mod, inp, out: _FM.__setitem__('fm', out.feature_maps))


def _maps(fms):
    hp, wp = DA_SIZE[0] // 14, DA_SIZE[1] // 14; out = []
    for fm in fms:
        if fm.dim() == 3: fm = fm[:, -hp * wp:].reshape(fm.shape[0], hp, wp, -1).permute(0, 3, 1, 2)
        out.append(fm.float())
    return out


@torch.no_grad()
def da_run(img):
    x = (F.interpolate(img, size=DA_SIZE, mode='bilinear', align_corners=False) - _MEAN) / _STD
    _FM.clear(); rel = da(pixel_values=x).predicted_depth
    rel = rel[:, None] if rel.dim() == 3 else rel
    rel = F.interpolate(rel.float(), size=(H, W), mode='bilinear', align_corners=False)
    fms = _FM.get('fm')
    if fms is None: fms = da.backbone(x).feature_maps
    return rel, _maps(fms)


class DAMaskHead(nn.Module):                             # A2와 같은 구조
    def __init__(self, c_da=384, c_red=32):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv2d(c_da, c_red, 1) for _ in range(4)])
        blk = lambda i, o, dl=1: [nn.Conv2d(i, o, 3, padding=dl, dilation=dl), nn.GroupNorm(8, o), nn.ReLU(True)]
        self.net = nn.Sequential(*blk(4 * c_red + 5, 64), *blk(64, 64, 2), *blk(64, 64, 4), *blk(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, img, d, feats):
        h, w = d.shape[-2:]; d1 = d[:, None]; v = (d1 > 0).float()
        f = [F.interpolate(r(x), size=(h, w), mode='bilinear', align_corners=False) for r, x in zip(self.red, feats)]
        return self.net(torch.cat(f + [img, d1, v], 1))[:, 0]


# ---------- 데이터: 우리 320 캐시 (원래 해상도 그대로, 늘리지 않음) ----------
SCALE = {1: 1000., 2: 4000.}
INTR = {1: np.load(f'{DATA}/camera_intrinsics/1-camIntrinsics-D435.npy'), 2: np.load(f'{DATA}/camera_intrinsics/2-camIntrinsics-L515.npy')}
KT = torch.tensor([[1., 1., 0., 0.]] + [[INTR[c][0, 0], INTR[c][1, 1], INTR[c][0, 2], INTR[c][1, 2]] for c in (1, 2)], device=dev)  # 1280x720 기준


class TCGB(torch.utils.data.Dataset):
    """train: 공식과 같이 depth 0~10 m, 뒤집기 + 색 증강 / test: 공식과 같이 1.5 m 넘는 값은 1.5로"""

    def __init__(self, split, train, color=True, folds=None):
        self.train, self.color, self.s = train, color, []
        for d in sorted(glob.glob(f'{DATA}/scene*'), key=lambda p: int(os.path.basename(p)[5:])):
            sm = json.load(open(f'{d}/metadata.json'))
            if sm.get('split') != split: continue
            fold = int(os.path.basename(d)[5:]) % 2                     # 장면 단위로 둘로 나눔 (같은 장면 프레임이 섞이지 않게)
            if folds is not None and fold not in folds: continue
            for cam, key in [(1, 'D435_valid_perspective_list'), (2, 'L515_valid_perspective_list')]:
                self.s += [(f'{d}/{v}', cam, fold) for v in sm[key] if os.path.exists(f'{d}/{v}/rgb{cam}.png')]

    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        d, cam, fold = self.s[i]
        rgb = cv2.cvtColor(cv2.imread(f'{d}/rgb{cam}.png'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        dep = cv2.imread(f'{d}/depth{cam}.png', cv2.IMREAD_ANYDEPTH).astype(np.float32) / SCALE[cam]
        gt = cv2.imread(f'{d}/depth{cam}-gt.png', cv2.IMREAD_ANYDEPTH).astype(np.float32) / SCALE[cam]
        for x in (dep, gt): x[~np.isfinite(x) | (x < 0)] = 0
        if self.train:
            dep[dep > 10] = 0; gt[gt > 10] = 0
        else:
            dep = np.minimum(dep, 1.5); gt = np.minimum(gt, 1.5)
        mk = cv2.imread(f'{d}/depth{cam}-gt-mask.png', cv2.IMREAD_GRAYSCALE) > 0
        if self.train:
            if self.color and np.random.rand() < 0.8:
                rgb = rgb * np.random.uniform(0.8, 1.2); mu = rgb.mean(); rgb = (rgb - mu) * np.random.uniform(0.8, 1.2) + mu
                rgb = rgb * np.random.uniform(0.95, 1.05, size=3).astype(np.float32)
                if np.random.rand() < 0.5: rgb = rgb + np.random.randn(*rgb.shape).astype(np.float32) * np.random.uniform(0, 0.03)
                rgb = np.clip(rgb, 0, 1).astype(np.float32)
            for ax in (0, 1):
                if np.random.rand() > 0.5: rgb, dep, gt, mk = (np.flip(x, ax) for x in (rgb, dep, gt, mk))
            rgb, dep, gt, mk = (np.ascontiguousarray(x) for x in (rgb, dep, gt, mk))
        return dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1), depth=torch.from_numpy(dep), gt=torch.from_numpy(gt),
                    mask=torch.from_numpy(mk), cam=cam, fold=fold)


def to_dev(b): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ---------- 마스크 도구 ----------
dil = lambda m, r: m if r == 0 else F.max_pool2d(m[:, None].float(), 2 * r + 1, 1, r)[:, 0] > 0.5
ero = lambda m, r: m if r == 0 else ~dil(~m, r)


def hysteresis(prob, hi=0.5, lo=0.1):
    p = prob.float().cpu().numpy(); out = np.zeros(p.shape, bool)
    for i in range(len(p)):
        weak = (p[i] > lo).astype(np.uint8)
        if not weak.any(): continue
        _, lab = cv2.connectedComponents(weak, connectivity=8)
        keep = np.unique(lab[(p[i] > hi) & (weak > 0)]); keep = keep[keep > 0]
        if len(keep): out[i] = np.isin(lab, keep)
    return torch.from_numpy(out).to(prob.device)


def drop_one(m, rng):
    out = m.clone().cpu().numpy()
    for i in range(len(out)):
        n_, lab = cv2.connectedComponents(out[i].astype(np.uint8), connectivity=8)
        if n_ > 1: out[i][lab == rng.randint(1, n_ - 1)] = False
    return torch.from_numpy(out).to(m.device)


def add_fp(m, valid, rng, k=2, r=10, gap=10):
    """마스크에서 gap px 넘게 떨어진, depth가 있는 곳에 반지름 r 원을 k개 (과검출 흉내, 320 기준 = A1의 640 기준 절반)"""
    out = m.cpu().numpy().astype(np.uint8).copy(); v = valid.cpu().numpy()
    for i in range(len(out)):
        cand = np.argwhere(v[i] & ~(cv2.dilate(out[i], np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0))
        canvas = np.ascontiguousarray(out[i])
        for _ in range(k):
            if len(cand): y, x = cand[rng.randint(0, len(cand) - 1)]; cv2.circle(canvas, (int(x), int(y)), int(r), 1, -1)
        out[i] = canvas
    return torch.from_numpy(out > 0).to(m.device)


def wobble(m, amp):
    """경계를 울퉁불퉁하게: 부드럽게 만든 마스크 + 저주파 잡음을 0.5에서 자름"""
    noise = F.interpolate(torch.randn(m.shape[0], 1, H // 16, W // 16, device=m.device), size=(H, W), mode='bilinear', align_corners=False)[:, 0]
    return (F.avg_pool2d(m[:, None].float(), 7, 1, 3)[:, 0] + amp * noise) > 0.5


def train_masks(gt_m, p, valid, mode, rng):
    """학습 때 ReMake에 넣을 마스크 (샘플마다 무작위). gt: 정답만 / mix: 섞기(이진) / soft: 섞기 + 예측은 확률 그대로"""
    n = gt_m.shape[0]
    if mode == 'gt': return gt_m.float(), ['gt'] * n
    out = torch.zeros(n, H, W, device=dev); kinds = []
    for i in range(n):
        u, g = rng.random(), gt_m[i:i + 1]
        if mode == 'lite': u = 0.1 if u < 0.5 else (0.5 if u < 0.9 else 0.9)   # 정답 50 / 예측 40 / 흔든 정답 10, 빈 마스크 없음
        if u < 0.05:
            m, k = torch.zeros_like(g, dtype=torch.float), 'empty'
        elif u < 0.30:
            m, k = g.float(), 'gt'
        elif u < 0.70:
            k = 'pred'
            if mode == 'soft':
                m = p[i:i + 1].float()
            else:
                m = (p[i:i + 1] > rng.choice([0.5, 0.3, 0.2, 0.1])) if rng.random() < 0.6 else hysteresis(p[i:i + 1], *rng.choice([(0.5, 0.1), (0.9, 0.05)]))
                if rng.random() < 0.3: r = rng.choice([1, 2]); m = dil(m, r) if rng.random() < 0.5 else ero(m, r)
                m = m.float()
        else:
            k, m = 'pert', g.clone()
            if rng.random() < 0.6: r = rng.choice([1, 2, 3]); m = dil(m, r) if rng.random() < 0.5 else ero(m, r)
            if rng.random() < 0.5: m = wobble(m, rng.uniform(0.1, 0.4))
            if rng.random() < 0.3: m = drop_one(m, rng)
            if rng.random() < 0.3: m = add_fp(m, valid[i:i + 1], rng, k=rng.choice([1, 2]), r=rng.randint(5, 15))
            m = m.float()
        out[i] = m[0]; kinds.append(k)
    return out, kinds


# ---------- 손실 ----------
_YS, _XS = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')


def normals(depth, cams):
    k = KT[cams]; sx, sy = W / 1280., H / 720.
    fx, fy, cx, cy = (k[:, 0] * sx)[:, None, None], (k[:, 1] * sy)[:, None, None], (k[:, 2] * sx)[:, None, None], (k[:, 3] * sy)[:, None, None]
    xyz = torch.stack([(_XS[None] - cx) * depth / fx, (_YS[None] - cy) * depth / fy, depth], 1)
    return get_surface_normal_from_xyz(xyz)


def tdc_loss(pred, gt, gt_m, cams):
    """TDCNet 공식: 투명 마스크 안(GT>0) L1 + 0.1 x 법선 (마스크 밖은 학습 안 함 → 배경은 선택으로 보존)"""
    zm = gt >= 0.01; lm = gt_m & zm
    zmd = ~(F.max_pool2d((~zm)[:, None].float(), 3, 1, 1)[:, 0] > 0.5); lmd = gt_m & zmd
    l1 = (pred - gt).abs()[lm].mean() if lm.any() else pred.sum() * 0
    with torch.no_grad(): ngt = normals(gt, cams)
    sn = (1 - F.cosine_similarity(normals(pred, cams), ngt, dim=1))[lmd].mean() if lmd.any() else pred.sum() * 0
    return l1 + BETA * sn, float(l1), float(sn)


# ---------- 평가 (320 그대로 + 256x144 공식 방식) ----------
ELL = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
REG = ['안쪽 띠', '바깥 띠', '주변', '배경']          # 320 기준 1~3 / 1~3 / 3~15 / >15 px (= 1단계 640 기준 1~6 / 1~6 / 6~30 / >30 px)


def regions(gt_m):
    out = []
    for m in gt_m.cpu().numpy().astype(np.uint8):
        b3, n15 = cv2.dilate(m, ELL(3)) > 0, cv2.dilate(m, ELL(15)) > 0; e3 = cv2.erode(m, ELL(3)) > 0; mb = m > 0
        out.append(np.stack([mb & ~e3, b3 & ~mb, n15 & ~b3, ~n15]))
    return torch.from_numpy(np.stack(out)).to(gt_m.device)


class Acc:
    def __init__(self): self.d = {}

    def add(self, name, key, vals, cams):
        for v, c in zip(vals, cams):
            if v == v: self.d.setdefault(name, {}).setdefault(key, {0: [], 1: [], 2: []}); self.d[name][key][0].append(v); self.d[name][key][c].append(v)

    def get(self, name, key, cam=0):
        v = self.d.get(name, {}).get(key, {}).get(cam, []); return float(np.mean(v)) if v else float('nan')


def score(acc, name, out, gt, gt_m, reg, cams, mask_in=None):
    t = gt_m & (gt > 0); e2 = (out - gt) ** 2; cnt = t.sum((1, 2)).float(); ok = cnt > 0
    r = torch.sqrt((e2 * t).sum((1, 2)) / cnt.clamp(min=1)) * 1000
    q = torch.maximum(out / gt.clamp(min=1e-8), gt / out.clamp(min=1e-8)); d = ((q < 1.05) & t).sum((1, 2)).float() / cnt.clamp(min=1) * 100
    acc.add(name, '투명 RMSE (320)', torch.where(ok, r, torch.nan).tolist(), cams)
    acc.add(name, '투명 δ1.05 (320)', torch.where(ok, d, torch.nan).tolist(), cams)
    rs = lambda x: F.interpolate(x[:, None].float(), size=(144, 256), mode='nearest')[:, 0]
    o_, g_, m_ = rs(out), rs(gt), rs(gt_m) > 0.5; mm = m_ & (g_ > 0); c_ = mm.sum((1, 2)).float(); ok_ = c_ > 0
    r256 = torch.sqrt((((o_ - g_) ** 2) * mm).sum((1, 2)) / c_.clamp(min=1)) * 1000
    q_ = torch.maximum(o_ / g_.clamp(min=1e-8), g_ / o_.clamp(min=1e-8)); d256 = ((q_ < 1.05) & mm).sum((1, 2)).float() / c_.clamp(min=1) * 100
    acc.add(name, '투명 RMSE (256x144)', torch.where(ok_, r256, torch.nan).tolist(), cams)
    acc.add(name, '투명 δ1.05 (256x144)', torch.where(ok_, d256, torch.nan).tolist(), cams)
    rg = (gt >= 0.3) & (gt <= 1.5)
    for j, rn in enumerate(REG):
        s = reg[:, j] & rg; cn = s.sum((1, 2)).float()
        acc.add(name, rn, torch.where(cn > 0, torch.sqrt((e2 * s).sum((1, 2)) / cn.clamp(min=1)) * 1000, torch.nan).tolist(), cams)
    if mask_in is not None:
        inter = (mask_in & gt_m).sum((1, 2)).float(); uni = (mask_in | gt_m).sum((1, 2)).float(); gs = gt_m.sum((1, 2)).float(); ps = mask_in.sum((1, 2)).float()
        acc.add(name, '마스크 IoU', torch.where(uni > 0, inter / uni.clamp(min=1), torch.nan).tolist(), cams)
        acc.add(name, '마스크 재현율', torch.where(gs > 0, inter / gs.clamp(min=1), torch.nan).tolist(), cams)
        acc.add(name, '마스크 정밀도', torch.where(ps > 0, inter / ps.clamp(min=1), torch.nan).tolist(), cams)


TEST = TCGB('test', False)
assert TEST[0]['rgb'].shape[-2:] == (H, W), f"320 캐시가 아님: {tuple(TEST[0]['rgb'].shape)} (ORIG_DIR는 비워 두세요)"
print(f'test {len(TEST)}장 (D435 {sum(c == 1 for _, c, _ in TEST.s)}, L515 {sum(c == 2 for _, c, _ in TEST.s)})')
KEYS = ['투명 RMSE (320)', '투명 δ1.05 (320)', '투명 RMSE (256x144)', '투명 δ1.05 (256x144)'] + REG + ['마스크 IoU', '마스크 재현율', '마스크 정밀도']
SHORT = ['투명320', 'δ320', '투명256', 'δ256', '안쪽띠', '바깥띠', '주변', '배경', 'IoU', '재현율', '정밀도']
head_full = DAMaskHead().to(dev).eval()


def report(acc, tag, path=None):
    res = {}
    for cam, cn in [(0, '전체'), (1, 'D435'), (2, 'L515')]:
        print(f'\n===== {tag} | {cn} (mm, %) =====\n{"":30s}' + ''.join(f'{s:>9s}' for s in SHORT))
        for name in acc.d:
            vals = [acc.get(name, k, cam) for k in KEYS]
            res.setdefault(name, {})[cn] = dict(zip(KEYS, [round(v, 3) for v in vals]))
            print(f'{name[:30]:30s}' + ''.join(f'{v:9.3f}' if k.startswith('마스크') else f'{v:9.2f}' for k, v in zip(KEYS, vals)))
    if path: json.dump(res, open(path, 'w'), indent=1, ensure_ascii=False)
    return res


# ---------- head들 ----------
def train_head(head, ds_train, path, tag):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dl = DataLoader(ds_train, HEAD_BS, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR); steps = HEAD_EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    pwt = torch.tensor(PW, device=dev); print(f'\n=== {tag}: train {len(ds_train)}장, {HEAD_EPOCHS} epoch ({el()}) ===')
    for ep in range(HEAD_EPOCHS):
        head.train(); t0, tl, n = time.time(), 0., 0
        for b in dl:
            b = to_dev(b); _, feats = da_run(b['rgb']); t = b['mask'].float()
            logit = head(b['rgb'], b['depth'], feats); p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pwt) + dice.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item(); n += 1
        print(f'  epoch {ep + 1}/{HEAD_EPOCHS} loss {tl / n:.4f} {(time.time() - t0) / 60:.1f}분')
    torch.save({'model_state_dict': head.state_dict(), 'epoch': HEAD_EPOCHS}, path); head.eval()


def load_or_train_head(path, ds_train, tag):
    h = DAMaskHead().to(dev)
    if os.path.exists(path) and torch.load(path, map_location='cpu', weights_only=False).get('epoch') == HEAD_EPOCHS:
        h.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']); print('head 불러옴:', path)
    else:
        train_head(h, ds_train, path, tag)
    return h.eval()


@torch.no_grad()
def head_iou(h, ds, every=10):
    tp = fp = fn = 0
    for b in DataLoader(Subset(ds, list(range(0, len(ds), every))), 32, num_workers=8):
        b = to_dev(b); _, feats = da_run(b['rgb']); pm = torch.sigmoid(h(b['rgb'], b['depth'], feats)) > 0.5; mk = b['mask']
        tp += int((pm & mk).sum()); fp += int((pm & ~mk).sum()); fn += int((~pm & mk).sum())
    return tp / max(tp + fp + fn, 1), tp / max(tp + fn, 1), tp / max(tp + fp, 1)


SEL = ['그대로', '선택 정답', '선택 head 0.5', '선택 head hyst', '선택 head hyst +2px', '선택 head hyst +4px', '선택p 0.05', '선택p 0.2']


@torch.no_grad()
def evaluate_free(model, indices, bs=32):
    """마스크를 안 받는 모델: 출력 그대로 + 여러 선택(마스크 밖이고 raw가 있으면 raw). 평가는 fp32"""
    acc = Acc(); model.eval()
    for b in DataLoader(Subset(TEST, indices), bs, shuffle=False, num_workers=8, pin_memory=True):
        b = to_dev(b); rgb, depth, gt = b['rgb'], b['depth'], b['gt']; gt_m = b['mask']; cams = b['cam'].tolist()
        rel, feats = da_run(rgb); p = torch.sigmoid(head_full(rgb, depth, feats)); hy = hysteresis(p, 0.5, 0.1)
        rv = depth > 0; reg = regions(gt_m)
        out = model(rgb, rel, depth).float().view_as(depth)
        keep = {'그대로': None, '선택 정답': gt_m, '선택 head 0.5': p > 0.5, '선택 head hyst': hy, '선택 head hyst +2px': dil(hy, 2),
                '선택 head hyst +4px': dil(hy, 4), '선택p 0.05': p >= 0.05, '선택p 0.2': p >= 0.2}
        for name, mk in keep.items():
            o = out if mk is None else torch.where(~mk & rv, depth, out)
            score(acc, name, o, gt, gt_m, reg, cams, mask_in=None if mk is None else mk)
    return acc


def train_tdc():
    d = f'{OUT}/{RUN}'; os.makedirs(d, exist_ok=True); final, last = f'{d}/final.tar', f'{d}/last.tar'
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = Net(VARIANT).to(dev)
    if os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location=dev, weights_only=False)['model_state_dict']); print('학습된 것 불러옴:', final); return model.eval()
    if OFFICIAL_AUG:
        from datasets.transcg import TransCG as TransCGOfficial
        ds_tr = TransCGOfficial(DATA, split='train', image_size=(320, 240), use_augmentation=True, rgb_augmentation_probability=0.8,
                                use_depth_augmentation=False, depth_min=0.0, depth_max=10.0, depth_norm=1.0, reldepth_model=None)
    else:
        ds_tr = TCGB('train', True)
    dl = DataLoader(ds_tr, BS, shuffle=True, drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=MILESTONES, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16); start = 0
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model_state_dict']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); start = ck['epoch']
        print(f'epoch {start}에서 이어서')
    print(f'\n=== TDCNet ({RUN}, fp32={FP32}, 공식 증강={OFFICIAL_AUG}) 학습: train {len(dl.dataset)}장 = {len(dl)} step/epoch, {EPOCHS} epoch, batch {BS}, lr {LR} ({el()}) ===')
    for ep in range(start, EPOCHS):
        model.train(); t0 = time.time(); s1 = s2 = 0.; n = 0
        for it, b in enumerate(dl):
            b = to_dev(b)
            if OFFICIAL_AUG:                                  # 공식 로더 키 → 우리 이름 (카메라는 fx로 구분)
                rgb, depth, gt, gt_m = b['rgb'], b['depth'], b['depth_gt'], b['depth_gt_mask'] > 0.5
                cams = torch.where((b['fx'].float() - KT[1, 0]).abs() < 1.0, 1, 2).long()
            else:
                rgb, depth, gt, gt_m, cams = b['rgb'], b['depth'], b['gt'], b['mask'], b['cam']
            rel = da_run(rgb)[0] if VARIANT == 'rel' else None
            with torch.autocast('cuda', dtype=AMP, enabled=not FP32): pred = model(rgb, rel, depth)
            loss, l1, sn = tdc_loss(pred.float().view_as(depth), gt, gt_m, cams)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            s1 += l1; s2 += sn; n += 1
            if it == 0 and ep == start: print(f'  첫 step L1 {l1 * 1000:.1f} mm 법선 {sn:.3f}')
        sched.step()
        msg = ''
        if (ep + 1) % 5 == 0 or ep + 1 == EPOCHS:
            a = evaluate_free(model, list(range(0, len(TEST), 50)))
            msg = f" | val(1/50) 투명256 그대로 {a.get('그대로', '투명 RMSE (256x144)'):.2f} / 선택 hyst {a.get('선택 head hyst', '투명 RMSE (256x144)'):.2f}"
        print(f'  epoch {ep + 1}/{EPOCHS} {(time.time() - t0) / 60:.1f}분 | L1 {s1 / n * 1000:.1f} mm 법선 {s2 / n:.3f} | lr {opt.param_groups[0]["lr"]:.0e}{msg} | 경과 {el()}')
        torch.save({'model_state_dict': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep + 1}, last)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': EPOCHS, 'config': dict(run=RUN, fp32=FP32, official_aug=OFFICIAL_AUG, epochs=EPOCHS, lr=LR, bs=BS)}, final)
    return model.eval()


def summary(res):
    print(f'\n\n################ B2 요약: TDCNet ({VARIANT}) + DA head 선택 (test 전체, 256x144, mm / %) ################')
    print(f'{"":24s}{"투명256":>9s}{"δ256":>8s}{"안쪽띠":>8s}{"바깥띠":>8s}{"주변":>8s}{"배경":>8s}{"D435":>8s}{"L515":>8s}')
    for k, v in res.items():
        a = v['전체']
        print(f'{k:24s}{a["투명 RMSE (256x144)"]:9.2f}{a["투명 δ1.05 (256x144)"]:8.2f}{a["안쪽 띠"]:8.2f}{a["바깥 띠"]:8.2f}{a["주변"]:8.2f}{a["배경"]:8.2f}'
              f'{v["D435"]["투명 RMSE (256x144)"]:8.2f}{v["L515"]["투명 RMSE (256x144)"]:8.2f}')
    print('\n--- 같은 평가 코드의 ReMake (b1_robust) ---')
    for nm in ['R0', 'R1', 'R2', 'R0L', 'R0L-lite']:
        f = f'{B1_DIR}/eval_{nm}.json'
        if not os.path.exists(f): continue
        r = json.load(open(f))
        for k in ['정답', 'head hyst | 선택', 'head hyst | 선택p']:
            if k in r: a = r[k]; print(f'ReMake {nm:9s} {k:18s}{a["전체"]["투명 RMSE (256x144)"]:8.2f}{a["전체"]["배경"]:8.2f}  D435 {a["D435"]["투명 RMSE (256x144)"]:.2f} L515 {a["L515"]["투명 RMSE (256x144)"]:.2f}')
    print('참고 (TDCNet 시기, 다른 평가 코드): TDCNet 단독 11.91 ± 0.16 (3 seed) / 개선판 12.53, 배경 11.2')


def dist_out(m):
    out = []
    for x in m.cpu().numpy().astype(np.uint8):
        out.append(cv2.distanceTransform(1 - x, cv2.DIST_L2, 5) if x.any() else np.full(x.shape, 1e3, np.float32))
    return torch.from_numpy(np.stack(out)).to(m.device)


HEAD_BASE = f'{TK}/b3_head/head_dav2-base.tar'   # B3에서 학습한 DA-V2-Base head (있으면 같이 평가)


def heads_for_eval():
    H_ = {'DA-V2-S': (lambda img: da_run(img)[1], head_full)}
    if os.path.exists(HEAD_BASE):
        mb = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Base-hf').to(dev).eval()
        for p_ in mb.parameters(): p_.requires_grad = False

        @torch.no_grad()
        def fb(img):
            x = (F.interpolate(img, size=DA_SIZE, mode='bilinear', align_corners=False) - _MEAN) / _STD
            return _maps(mb.backbone(x).feature_maps)
        hb = DAMaskHead(c_da=768).to(dev); hb.load_state_dict(torch.load(HEAD_BASE, map_location='cpu', weights_only=False)['model_state_dict']); hb.eval()
        H_['DA-V2-B'] = (fb, hb)
    print('평가할 head:', list(H_))
    return H_


@torch.no_grad()
def eval_soft(model, indices, bs=32):
    """최종안 후보: head hyst 마스크로부터의 거리에 따라 예측↔raw를 섞음 (B2-b·B3와 같은 정의), head 2종"""
    acc = Acc(); model.eval(); HS = heads_for_eval()
    for b in DataLoader(Subset(TEST, indices), bs, shuffle=False, num_workers=8, pin_memory=True):
        b = to_dev(b); rgb, depth, gt = b['rgb'], b['depth'], b['gt']; gt_m = b['mask']; cams = b['cam'].tolist()
        rv = depth > 0; reg = regions(gt_m); out = model(rgb, None, depth).float().view_as(depth)
        for hn, (ff, hd) in HS.items():
            hy = hysteresis(torch.sigmoid(hd(rgb, depth, ff(rgb))), 0.5, 0.1); dist = dist_out(hy)
            for name, r1, r2 in [('부드럽게 2→6px', 2, 6), ('부드럽게 4→12px', 4, 12)]:
                w = torch.where(rv, ((r2 - dist) / (r2 - r1)).clamp(0, 1), torch.ones_like(dist))
                score(acc, f'{hn} | {name}', w * out + (1 - w) * depth, gt, gt_m, reg, cams, mask_in=hy)
    return acc


# ==================== 실행 ====================
try:
    head_full = load_or_train_head(HEAD_FULL, TCGB('train', True, color=False), 'test용 head (train 전체)')
    if OFFICIAL_AUG:
        from datasets.transcg import TransCG as TransCGOfficial
        _o = TransCGOfficial(DATA, split='train', image_size=(320, 240), use_augmentation=True, rgb_augmentation_probability=0.8, use_depth_augmentation=False,
                             depth_min=0.0, depth_max=10.0, depth_norm=1.0, reldepth_model=None)[0]
        print(f"공식 로더 확인: rgb {tuple(_o['rgb'].shape)} {float(_o['rgb'].min()):.2f}~{float(_o['rgb'].max()):.2f}, depth 최대 {float(_o['depth'].max()):.2f} m, fx {float(_o['fx']):.1f} (D435 {float(KT[1, 0]):.1f} / L515 {float(KT[2, 0]):.1f})")
    t0 = time.time(); _m = Net(VARIANT).to(dev); _b = to_dev(next(iter(DataLoader(TCGB('train', True), 4, shuffle=True))))
    with torch.autocast('cuda', dtype=AMP): _p = _m(_b['rgb'], da_run(_b['rgb'])[0], _b['depth'])
    _l = tdc_loss(_p.float().view_as(_b['depth']), _b['gt'], _b['mask'], _b['cam']); _l[0].backward()
    _ = evaluate_free(_m, list(range(0, 64))); del _m
    print(f'스모크 테스트 통과 (loss {_l[0].item():.4f}) {(time.time() - t0) / 60:.1f}분')
    model = train_tdc()
    path = f'{OUT}/eval_{RUN}.json'
    print(f'\n=== 평가: test 전체 {len(TEST)}장 ({el()}) ===')
    res = report(evaluate_free(model, list(range(len(TEST)))), f'B2 TDCNet {VARIANT}', path)
    summary(res)
    res_soft = report(eval_soft(model, list(range(len(TEST)))), f'B4 {RUN} 부드러운 합성', f'{OUT}/eval_{RUN}_soft.json')
    print(f'\n끝. 결과: {OUT} | 경과 {el()}')
except Exception:
    traceback.print_exc(); print('!!! B2 중단')
