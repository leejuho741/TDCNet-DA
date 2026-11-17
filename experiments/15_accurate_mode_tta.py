# ============================================================
# 30. 성능 위주 + 좌우 TTA (8번 표 한 줄 추가용). L4 약 20~25분 (준비 포함)
#   22번과 같은 TransCG test 1/4, 같은 표준 지표 + 배경 영역 / 속도는 29번과 같은 fp16 + CUDA graph, batch 1
#   행: 속도 위주 / 성능 위주 / 속도 위주 + TTA / 성능 위주 + TTA
#   TTA = 깊이 모델 출력을 원본과 좌우 뒤집기로 평균 (23번 ④와 같음). 마스크는 원본 한 번
#   결과: MyDrive/tdc_finetune/next30_tta/ (tta.json, log.txt)
# ============================================================
import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import os; os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
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
VARIANT = 'plain'                 # 'plain': TDCNet 그대로 / 'rel': 상대 depth(DA)를 인코더 4단계에 0-초기화로 더함 (다른 런타임에서 따로)
EPOCHS, BS, LR, MILESTONES = 40, 8, 1e-3, [35]   # TDCNet 공식 레시피 (scratch, AdamW, epoch 35에서 x0.1)
BETA = 0.1                        # 법선 loss 가중치 (공식)
HEAD_EPOCHS, HEAD_BS, HEAD_LR, PW = 10, 16, 1e-3, 10.0   # head를 불러올 때 확인용 (B1과 같음)
H, W = 240, 320
DA_SIZE = (252, 336)
AUTO_SHUTDOWN = False
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


if not isinstance(sys.stdout, _Tee): sys.stdout = _Tee(sys.stdout, open(f'{OUT}/log_speed_compare.txt', 'a'))
el = lambda: f'{(time.time() - T_START) / 3600:.2f}h'


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



# ==================== 22. 같은 환경에서 SOTA 비교 + 제대로 된 일반화 평가 ====================
#   공식 체크포인트가 공개된 방법을 전부 같은 입력·같은 지표·같은 GPU로 다시 돌림
#     DFNet (TransCG 공식) / TODE-Trans (공식, TransCG 학습) / FDCT (공식 TransCG) / TDCNet (공식 레시피 재현) / ReMake (공식, 정답 마스크·head 마스크)
#     우리: 처음부터 s0 단독 / 최종 실시간 (기존 head + 게이트) / 최종 정밀 (중간 해상도 head + 게이트)
#   표준 지표 (TransCG 논문 방식, 정답 투명 마스크 안, 이미지별 평균): RMSE, REL, MAE, δ1.05, δ1.10, δ1.25 + 우리 영역 지표(경계·배경)
#   데이터: TransCG test 1/4 (5,881장) / ClearGrasp real-test D435·D415 / ClearGrasp 합성 known(synthetic-test)·novel(synthetic-val) 각 최대 300장
#   일반화는 전부 '적응 없이'. 우리 적응(C) 결과는 따로 표시
#   속도: 모든 방법 같은 조건 (batch 1, fp32, 그래프 없이) 중앙값
#   결과: MyDrive/tdc_finetune/next22_sota/ (sota.json, log.txt)
import copy
N13 = f'{TK}/next13_general'; N22 = f'{TK}/next30_tta'; os.makedirs(N22, exist_ok=True)
sys.stdout = _Tee(sys.stdout.s if isinstance(sys.stdout, _Tee) else sys.stdout, open(f'{N22}/log.txt', 'a'))
print(f'\n######## 30. 성능 위주 + TTA 시작 {time.strftime("%m-%d %H:%M")} | {torch.cuda.get_device_name()}')
B_EPOCHS, B_BS, B_LR = 3, 8, 1e-3
C_STEPS, C_BS, C_LR, W_OUT, W_CONS, W_ANCHOR = 600, 8, 2e-4, 1.0, 0.5, 1.0
torch.manual_seed(0); random.seed(0); np.random.seed(0); rng = random.Random(0)

CG = '/content/cleargrasp'
if False:   # 30번은 ClearGrasp 필요 없음
    os.makedirs(CG, exist_ok=True)
    run(f'wget -q -O /content/cg.tar https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar')
    run(f"tar -xf /content/cg.tar -C {CG} --wildcards '*real-test*' '*real-val*' && rm /content/cg.tar")
RGBS = sorted(f for f in glob.glob(f'{CG}/**/*-transparent-rgb-img.jpg', recursive=True) if 'real' in f)
print(f'ClearGrasp 실제 이미지 {len(RGBS)}장. 폴더 예: {sorted({os.path.dirname(f) for f in RGBS})[:6]}')
if RGBS: print('한 장의 파일들:', sorted(os.listdir(os.path.dirname(RGBS[0])))[:8])


def read_exr(p):
    x = cv2.imread(p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if x is None:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(p); dw = f.header()['dataWindow']; sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        ch = list(f.header()['channels'].keys())[0]
        x = np.frombuffer(f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(sz)
    if x.ndim == 3: x = x[..., 0]
    x = x.astype(np.float32); x[~np.isfinite(x) | (x < 0)] = 0; return x


class CGDS(torch.utils.data.Dataset):
    def __init__(self, files): self.f = files

    def __len__(self): return len(self.f)

    def __getitem__(self, i):
        p = self.f[i]; pre = p[:-len('-transparent-rgb-img.jpg')]
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        dep = read_exr(pre + '-transparent-depth-img.exr'); gt = read_exr(pre + '-opaque-depth-img.exr')
        mk = cv2.imread(pre + '-mask.png', cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
        dep, gt = (np.minimum(cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST), 1.5) for x in (dep, gt))
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 127
        cam = 1 if 'd435' in p.lower() else 2
        return dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1), depth=torch.from_numpy(dep), gt=torch.from_numpy(gt), mask=torch.from_numpy(mk), cam=cam)



VAL = [f for f in RGBS if '/real-val/' in f]; TST = [f for f in RGBS if '/real-test/' in f]
print(f'real-val {len(VAL)}장 (적응·선택용) / real-test {len(TST)}장 (평가용)')
REG = ['경계', '안쪽 띠', '바깥 띠', '주변', '배경']


def regions(gt_m):
    out = []
    for m in gt_m.cpu().numpy().astype(np.uint8):
        b3, n15 = cv2.dilate(m, ELL(3)) > 0, cv2.dilate(m, ELL(15)) > 0; e3 = cv2.erode(m, ELL(3)) > 0; mb = m > 0
        inn, outb = mb & ~e3, b3 & ~mb
        out.append(np.stack([inn | outb, inn, outb, n15 & ~b3, ~n15]))
    return torch.from_numpy(np.stack(out)).to(gt_m.device)


KEYS = ['투명 RMSE (256x144)', '투명 δ1.05 (256x144)'] + REG + ['마스크 IoU']
SHORT = ['투명', 'δ1.05', '경계', '안쪽띠', '바깥띠', '주변', '배경', 'IoU']
dab = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Base-hf').to(dev).eval()
for p_ in dab.parameters(): p_.requires_grad = False
hb = DAMaskHead(c_da=768).to(dev); hb.load_state_dict(torch.load(f'{TK}/b3_head/head_dav2-base.tar', map_location='cpu', weights_only=False)['model_state_dict']); hb.eval()
for p_ in hb.parameters(): p_.requires_grad = False
norm = lambda x, size: (F.interpolate(x, size=size, mode='bilinear', align_corners=False) - _MEAN) / _STD
da_feats = lambda rgb: _maps(dab.backbone(norm(rgb, DA_SIZE)).feature_maps)
fl = lambda x: torch.flip(x, [-1])


class FNet(nn.Module):
    """TDCNet + (선택) DA 특징 주입. 특징 i(3·6·9·12층) → 인코더 단계 i. 18x24에서 768→64→c로 줄인 뒤 단계 해상도로 키워 더함"""
    def __init__(self, use_feat, c_da=768, c_mid=64):
        super().__init__(); self.use_feat = use_feat; self.base = TDCNet(lambda_val=1, res=True)
        if use_feat:
            self.base.eval()
            with torch.no_grad(): chs = [e.shape[1] for e in self.base.encoder(torch.zeros(1, 4, H, W))]
            self.base.train(); self.proj = nn.ModuleList()
            for c in chs:
                m = nn.Sequential(nn.Conv2d(c_da, c_mid, 1), nn.GroupNorm(8, c_mid), nn.GELU(), nn.Conv2d(c_mid, c, 3, padding=1))
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); self.proj.append(m)

    def forward(self, img, depth, feats=None):
        if not self.use_feat: return self.base(img, depth)
        n, h, w = depth.shape; d = depth.view(n, 1, h, w)
        enc = self.base.encoder(torch.cat((img, d), 1)); res = self.base.resnet(d)
        for i in range(4):
            enc[i] = enc[i] + res[i] + F.interpolate(self.proj[i](feats[i]), size=enc[i].shape[-2:], mode='bilinear', align_corners=False)
        return self.base.final(self.base.decoder(enc))


def load_repro(net):
    sd = torch.load(f'{OUT}/repro_fp32/final.tar', map_location='cpu', weights_only=False)['model_state_dict']
    miss, extra = net.load_state_dict(sd, strict=False)
    assert all(k.startswith('proj') for k in miss) and not extra, (miss[:5], extra[:5])



tdc0 = Net('plain').to(dev); load_repro(tdc0); tdc0.eval()
adp = FNet(True).to(dev); adp.load_state_dict(torch.load(f'{TK}/next7_feat/adapter/final.tar', map_location=dev, weights_only=False)['model_state_dict']); adp.eval()
DISKS = {r: (((torch.arange(-r, r + 1)[:, None] ** 2 + torch.arange(-r, r + 1)[None] ** 2) <= r * r).float().to(dev)[None, None]) for r in range(1, 13)}


def hyst_gpu(p, hi=0.5, lo=0.1, iters=48):
    weak = (p > lo).float()[:, None]; m = (p > hi).float()[:, None]
    for _ in range(iters): m = F.max_pool2d(m, 3, 1, 1) * weak
    return m[:, 0] > 0.5


def dist_gpu(m):
    d = torch.full(m.shape, 13., device=m.device); mf = m[:, None].float()
    for r in range(12, 0, -1):
        d = torch.where(F.conv2d(mf, DISKS[r], padding=r)[:, 0] > 0.5, torch.full_like(d, float(r)), d)
    return torch.where(m, torch.zeros_like(d), d)


def gate(out, depth, m):
    dd = dist_gpu(m); w = torch.where(depth > 0, ((12 - dd) / 8).clamp(0, 1), torch.ones_like(dd)); return w * out + (1 - w) * depth




def freeze_base(net):
    for n_, p_ in net.named_parameters(): p_.requires_grad = n_.startswith('proj')
    net.train(); net.base.eval(); return net


def out_of(model, rgb, depth, f0, f1=None):
    """model: ('tdc', net) 또는 ('fnet', net). f1이 있으면 좌우 TTA"""
    kind, net = model
    run_ = (lambda r, d, f: net(r, None, d)) if kind == 'tdc' else (lambda r, d, f: net(r, d, f))
    o = run_(rgb, depth, f0).float().view_as(depth)
    if f1 is not None: o = (o + fl(run_(fl(rgb), fl(depth), f1).float().view_as(depth))) / 2
    return o


def mask_of(p, cfg): hi, lo, er = cfg; return ero(hyst_gpu(p, hi, lo), er)


CG_CACHE = {}


@torch.no_grad()
def cg_batches(files, key):
    """ClearGrasp 배치를 DA 특징·head 확률과 함께 GPU 메모리에 한 번만 만들어 둠 (286장이라 작음)"""
    if key not in CG_CACHE:
        L = []
        for b in DataLoader(CGDS(files), 16, shuffle=False, num_workers=8):
            b = to_dev(b); f0, f1 = da_feats(b['rgb']), da_feats(fl(b['rgb']))
            b['f0'], b['f1'] = f0, f1; b['p'] = torch.sigmoid(hb(b['rgb'], b['depth'], f0)).float(); b['reg'] = regions(b['mask']); L.append(b)
        CG_CACHE[key] = L
    return CG_CACHE[key]


@torch.no_grad()
def cg_eval(acc, name, model, files, key, cfg=(0.5, 0.1, 2), tta=False, gate_on=True):
    for b in cg_batches(files, key):
        o = out_of(model, b['rgb'], b['depth'], b['f0'], b['f1'] if tta else None)
        m = mask_of(b['p'], cfg)
        score(acc, name, gate(o, b['depth'], m) if gate_on else o, b['gt'], b['mask'], b['reg'], b['cam'].tolist(), mask_in=m if gate_on else None)


@torch.no_grad()
def cg_full_rmse(model, files, key, cfg, tta=False):
    """설정 고르기용 한 숫자: real-val 전체 이미지(정답 0.3~1.5 m) RMSE. 투명 영역만 보지 않음 (투명·경계·배경을 같이 봄)"""
    s = n = 0.
    for b in cg_batches(files, key):
        o = gate(out_of(model, b['rgb'], b['depth'], b['f0'], b['f1'] if tta else None), b['depth'], mask_of(b['p'], cfg))
        v = (b['gt'] >= 0.3) & (b['gt'] <= 1.5); s += float(((o - b['gt']) ** 2)[v].sum()); n += int(v.sum())
    return np.sqrt(s / max(n, 1)) * 1000


@torch.no_grad()
def adabn(model, files, key):
    """A1: BatchNorm 통계만 새 데이터(정답 없는 이미지)로 다시 계산"""
    kind, net = model; net = copy.deepcopy(net)
    for mod in net.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.reset_running_stats(); mod.momentum = None
    net.train()
    for _ in range(2):
        for b in cg_batches(files, key):
            for r_, d_, f_ in [(b['rgb'], b['depth'], b['f0']), (fl(b['rgb']), fl(b['depth']), b['f1'])]:
                (net(r_, None, d_) if kind == 'tdc' else net(r_, d_, f_))
    net.eval(); return (kind, net)




# ---------- 표준 지표 ----------
class StdAcc:
    """TransCG 논문 방식: 정답 투명 마스크(정답 depth 유효) 안에서 이미지별로 계산해 평균"""
    def __init__(self): self.d = {}

    def add(self, name, out, gt, gm, cams, camname):
        for j in range(out.shape[0]):
            sel = gm[j] & (gt[j] > 0.01)
            if int(sel.sum()) < 10: continue
            o, g_ = out[j][sel].float(), gt[j][sel].float(); e = o - g_; r = torch.maximum(o.clamp(min=1e-6) / g_, g_ / o.clamp(min=1e-6))
            v = [float(e.pow(2).mean().sqrt()) * 1000, float((e.abs() / g_).mean()) * 100, float(e.abs().mean()) * 1000,
                 float((r < 1.05).float().mean()) * 100, float((r < 1.10).float().mean()) * 100, float((r < 1.25).float().mean()) * 100]
            for k in ('전체', camname[cams[j]]): self.d.setdefault(name, {}).setdefault(k, []).append(v)

    def table(self, title, names, camkeys):
        print(f'\n################ {title} ################')
        print(f'{"":36s}{"RMSE":>8s}{"REL%":>8s}{"MAE":>8s}{"δ1.05":>8s}{"δ1.10":>8s}{"δ1.25":>8s}' + ''.join(f'{"RMSE " + c:>11s}' for c in camkeys) + f'{"장수":>6s}')
        out = {}
        for n_ in names:
            if n_ not in self.d: print(f'{n_[:36]:36s}  (결과 없음)'); continue
            m = np.mean(self.d[n_]['전체'], 0); out[n_] = {k: np.mean(v, 0).tolist() for k, v in self.d[n_].items()}
            print(f'{n_[:36]:36s}' + ''.join(f'{x:8.2f}' for x in m) + ''.join(f'{np.mean(self.d[n_][c], 0)[0]:11.2f}' if c in self.d[n_] else f'{"—":>11s}' for c in camkeys)
                  + f'{len(self.d[n_]["전체"]):6d}')
        return out


ALLRES = {}
RUN_B = False; RUN_SPEED = False   # 22에서 끝남

# ---------- 다른 저장소 모델을 서로 안 섞이게 불러오기 (models·utils·datasets 이름이 겹침) ----------
def isolated_import(repo, fn):
    keep = {k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'utils', 'datasets')}
    for k in keep: sys.modules.pop(k)
    sys.path.insert(0, repo); cwd = os.getcwd(); os.chdir(repo)
    try: return fn()
    finally:
        for k in [k for k in sys.modules if k.split('.')[0] in ('models', 'utils', 'datasets')]: sys.modules.pop(k)
        sys.modules.update(keep); sys.path.remove(repo); os.chdir(cwd)


def build_from_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    try:
        from utils.builder import ConfigBuilder
        m = ConfigBuilder(**cfg).get_model()
    except Exception as e:
        print('   ConfigBuilder 실패 → 직접 생성:', type(e).__name__, str(e)[:120])
        mt = cfg['model']['type']; mod = importlib.import_module(f'models.{mt}'); m = getattr(mod, mt)(**cfg['model'].get('params', {}))
    def find(d, key):
        if isinstance(d, dict):
            if key in d: return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None: return r
        return None
    dn = find(cfg.get('dataset', cfg), 'depth_norm'); print(f'   설정: depth_norm {dn}, depth_min {find(cfg, "depth_min")}, depth_max {find(cfg, "depth_max")}, image_size {find(cfg, "image_size")}')
    PRE_LAST[0] = (find(cfg.get('dataset', cfg), 'depth_min'), find(cfg.get('dataset', cfg), 'depth_max'))
    return m, float(dn or 1.0)


def load_sd(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
    sd = {k.replace('module.', ''): v for k, v in sd.items() if torch.is_tensor(v)}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'   가중치: 빠진 키 {len(miss)} / 남는 키 {len(unexp)}' + (f' 예: {miss[:3]} {unexp[:3]}' if miss or unexp else ''))
    return len(miss) == 0


EXT = {}   # 이름 → (모델, depth_norm)
PRE = {}   # 이름 → (depth_min, depth_max): 공식 설정의 입력 depth 범위
PRE_LAST = [None]
SOTAD = f'{TK}/sota'; os.makedirs(SOTAD, exist_ok=True)
# DFNet (TransCG 공식)
try:
    if not os.path.isdir('/content/TransCG'): run('git clone -q https://github.com/Galaxies99/TransCG.git /content/TransCG')
    ck = f'{SOTAD}/dfnet_checkpoint.tar'
    if not os.path.exists(ck): run(f'pip -q install gdown && gdown -q 1oZi9zdOg0WYuTHM10xlyq5FRlfoKDKzU -O "{ck}"')
    if not os.path.exists(ck): print('!! DFNet 체크포인트 없음. 브라우저로 https://drive.google.com/file/d/1oZi9zdOg0WYuTHM10xlyq5FRlfoKDKzU/view 를 받아', ck, '로 넣어 주세요')
    cfgs = sorted(glob.glob('/content/TransCG/configs/*.yaml')); print('DFNet 설정 후보:', [os.path.basename(c) for c in cfgs])
    cfgp = next((c for c in cfgs if 'default' in c), next((c for c in cfgs if 'inference' in c), cfgs[0]))
    m, dn = isolated_import('/content/TransCG', lambda: build_from_cfg(cfgp)); print(f'DFNet 설정 {os.path.basename(cfgp)}, depth_norm {dn}')
    if load_sd(m, ck): EXT['DFNet (TransCG 공식)'] = (m.to(dev).eval(), dn); PRE['DFNet (TransCG 공식)'] = PRE_LAST[0]
except Exception: traceback.print_exc(); print('!! DFNet 준비 실패 → 건너뜀')
# TODE-Trans (공식, TransCG 학습)
try:
    if not os.path.isdir('/content/TODE'): run('git clone -q https://github.com/yuchendoudou/TODE.git /content/TODE')
    run('pip -q install timm einops')
    cks = [c for c in glob.glob('/content/TODE/checkpoints/**/*', recursive=True) if c.endswith(('.tar', '.pth', '.pt'))]; print('TODE 체크포인트 후보:', cks)
    cfgs = sorted(glob.glob('/content/TODE/configs/*.yaml')); print('TODE 설정 후보:', [os.path.basename(c) for c in cfgs])
    ck = next((c for c in cks if 'transcg' in c.lower()), cks[0] if cks else None)
    cfgp = next((c for c in cfgs if os.path.basename(c).startswith('train_transcg')), next((c for c in cfgs if 'transcg' in c.lower()), cfgs[0]))
    m, dn = isolated_import('/content/TODE', lambda: build_from_cfg(cfgp)); print(f'TODE 설정 {os.path.basename(cfgp)}, 체크포인트 {ck}, depth_norm {dn}')
    if ck and load_sd(m, ck): EXT['TODE-Trans (공식)'] = (m.to(dev).eval(), dn); PRE['TODE-Trans (공식)'] = PRE_LAST[0]
except Exception: traceback.print_exc(); print('!! TODE 준비 실패 → 건너뜀')
# FDCT
FDCT_REPO = '/content/FDCT'
if not os.path.isdir(FDCT_REPO): run(f'git clone -q https://github.com/Nonmy/FDCT.git {FDCT_REPO}')
sys.path.insert(0, FDCT_REPO); from Model import FDCT; sys.path.remove(FDCT_REPO)
fdct = FDCT().to(dev).eval(); FDCT_OK = False
cands = sorted({f for pat in ['/content/drive/MyDrive/checkpoints/**/*', '/content/drive/MyDrive/Colab Notebooks/FDCT/**/*', f'{TK}/fdct/**/*']
                for f in glob.glob(pat, recursive=True) if f.lower().endswith(('.tar', '.pth', '.pt', '.ckpt')) and os.path.isfile(f)}, key=len)
cands = [c for c in cands if 'transcg' in os.path.basename(c).lower()]
msd = fdct.state_dict()
for c in cands:
    sd = torch.load(c, map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
    sd = {k.replace('module.', ''): v for k, v in sd.items() if torch.is_tensor(v)}
    if set(sd) == set(msd): fdct.load_state_dict(sd); FDCT_OK = True
    else:
        a, b_ = list(sd.items()), list(msd.items())
        if len(a) == len(b_) and all(x[1].shape == y[1].shape for x, y in zip(a, b_)):
            fdct.load_state_dict({k: v for (k, _), (_, v) in zip(b_, a)}); FDCT_OK = True
    if FDCT_OK: print('FDCT 체크포인트:', c); break
if not FDCT_OK: print('!! FDCT TransCG 체크포인트를 못 찾음 → FDCT는 건너뜀')

if FDCT_OK: EXT['FDCT (공식 TransCG)'] = (fdct, 1.0); PRE['FDCT (공식 TransCG)'] = (0.3, 1.5)   # FDCT는 TransCG 공식 로더 기본값


def ext_forward(m, dn, rgb, depth):
    o = m(rgb, depth / dn)
    o = o[0] if isinstance(o, (tuple, list)) else o
    return (o.float() * dn).reshape(depth.shape)


# ---------- 우리 모델 ----------
def loadF(path):
    n_ = FNet(True).to(dev); n_.load_state_dict(torch.load(path, map_location=dev, weights_only=False)['model_state_dict']); return n_.eval()


sc0 = loadF(f'{TK}/next12_scratch/s0/final.tar')
MID = (378, 504)
def _maps_any(fms, size):
    hp, wp = size[0] // 14, size[1] // 14
    return [(fm[:, -hp * wp:].reshape(fm.shape[0], hp, wp, -1).permute(0, 3, 1, 2) if fm.dim() == 3 else fm).float() for fm in fms]
da_mid = lambda rgb: _maps_any(dab.backbone(norm(rgb, MID)).feature_maps, MID)
hmid = None
if os.path.exists(f'{TK}/next20_hires2/head_mid.tar'):
    hmid = DAMaskHead(c_da=768).to(dev); hmid.load_state_dict(torch.load(f'{TK}/next20_hires2/head_mid.tar', map_location='cpu', weights_only=False)['model_state_dict']); hmid.eval()
print('공식 입력 범위:', PRE); print('비교 대상 외부 모델:', list(EXT), '| 중간 해상도 head:', hmid is not None)


def all_outputs(rgb, depth, with_mid=True):
    """이 배치에 대한 모든 방법의 출력 (320x240, m 단위)과 우리 게이트 마스크"""
    O = {}
    for nm, (m, dn) in EXT.items():
        try:
            O[nm] = ext_forward(m, dn, rgb, depth)
            if PRE.get(nm) and None not in PRE[nm]:
                lo_, hi_ = PRE[nm]; d_ = torch.where(depth < lo_, torch.zeros_like(depth), depth.clamp(max=hi_))
                O[nm + ' · 공식 전처리'] = ext_forward(m, dn, rgb, d_)
        except Exception as e:
            if not hasattr(all_outputs, 'warned'): all_outputs.warned = set()
            if nm not in all_outputs.warned: print(f'!! {nm} 실행 실패: {type(e).__name__} {str(e)[:150]}'); all_outputs.warned.add(nm)
    O['TDCNet (공식 레시피 재현)'] = tdc0(rgb, None, depth).float().view_as(depth)
    f0 = da_feats(rgb); p = torch.sigmoid(hb(rgb, depth, f0)).float(); m = ero(hyst_gpu(p), 2)
    D = sc0(rgb, depth, f0).float().view_as(depth)
    O['우리: 깊이 모델 단독'] = D; O['우리 최종 (실시간, 67 FPS)'] = gate(D, depth, m)
    if with_mid and hmid is not None:
        mm = ero(hyst_gpu(torch.sigmoid(hmid(rgb, depth, da_mid(rgb))).float()), 2); O['우리 최종 (정밀, 44 FPS)'] = gate(D, depth, mm)
    return O, p, m


ORDER = ['DFNet (TransCG 공식)', 'DFNet (TransCG 공식) · 공식 전처리', 'TODE-Trans (공식)', 'TODE-Trans (공식) · 공식 전처리', 'FDCT (공식 TransCG)', 'FDCT (공식 TransCG) · 공식 전처리', 'TDCNet (공식 레시피 재현)', 'ReMake (정답 마스크)', 'ReMake (우리 head 마스크)',
         '우리: 깊이 모델 단독', '우리 최종 (실시간, 67 FPS)', '우리 최종 (정밀, 44 FPS)']


import gc
RESULTS = {}

def timeit(fn, warm=30, n=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t_ = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t_) * 1000)
    return float(np.median(ts))


def cap(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st); g_ = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_): fn()
    return lambda: g_.replay()


# ==================== 30. 성능 위주 + TTA ====================
import gc
for _n in ['remake', 'reldepth', 'ds']:
    if _n in globals(): del globals()[_n]
gc.collect(); torch.cuda.empty_cache()
assert hmid is not None, '중간 해상도 head(next20_hires2/head_mid.tar)가 없음'


def ours_out(rgb, depth, mid, tta):
    fe = da_feats(rgb); D = sc0(rgb, depth, fe).float().view_as(depth)
    if tta: D = (D + fl(sc0(fl(rgb), fl(depth), da_feats(fl(rgb))).float().view_as(depth))) / 2
    p_ = torch.sigmoid(hmid(rgb, depth, da_mid(rgb))).float() if mid else torch.sigmoid(hb(rgb, depth, fe)).float()
    return gate(D, depth, ero(hyst_gpu(p_), 2))


ROWS = [('속도 위주', False, False), ('성능 위주', True, False), ('속도 위주 + TTA', False, True), ('성능 위주 + TTA', True, True)]
EVERY = 4; SUB = list(range(0, len(TEST), EVERY)); std = StdAcc(); acc = Acc(); t0 = time.time(); i0 = 0
with torch.no_grad():
    for bi, b in enumerate(DataLoader(Subset(TEST, SUB), 16, shuffle=False, num_workers=8, pin_memory=True)):
        b = to_dev(b); rgb, depth, gt, gm = b['rgb'], b['depth'], b['gt'], b['mask']; cams = b['cam'].tolist(); i0 += len(cams)
        reg = regions(gm)
        for nm, mid, tta in ROWS:
            o = ours_out(rgb, depth, mid, tta)
            std.add(nm, o, gt, gm, cams, {1: 'D435', 2: 'L515'}); score(acc, nm, o, gt, gm, reg, cams)
        if bi % 100 == 0: print(f'  [TransCG 1/4] {i0}/{len(SUB)} {(time.time() - t0) / 60:.1f}분')
RT = std.table(f'30. TransCG test 1/{EVERY} — 표준 지표 (22번과 같은 방식)', [r[0] for r in ROWS], ['D435', 'L515'])
RA = report(acc, '30. 영역 지표')

s0_ = to_dev({k: (v[None] if torch.is_tensor(v) else torch.tensor([v])) for k, v in TEST[0].items()}); r_, d_ = s0_['rgb'].contiguous(), s0_['depth'].contiguous()
SPD = {}
with torch.no_grad():
    for nm, mid, tta in ROWS:
        def f16(mid=mid, tta=tta):
            with torch.autocast('cuda', dtype=torch.float16): return ours_out(r_, d_, mid, tta)
        try: ms = timeit(cap(f16), 30, 100); how = 'fp16 + CUDA graph'
        except Exception as e:
            print(f'  {nm}: CUDA graph 실패 ({type(e).__name__}) → fp16 eager'); torch.cuda.synchronize(); ms = timeit(f16, 30, 100); how = 'fp16 eager'
        SPD[nm] = {'ms': ms, 'fps': 1000 / ms, 'how': how}

print('\n################ 30. 8번 표용 요약 (TransCG test 1/4) ################')
print(f'{"모드":20s}{"RMSE":>8s}{"δ1.05":>8s}{"배경":>8s}{"D435":>8s}{"L515":>8s}{"FPS":>7s}')
OUTJ = {}
for nm, _, _ in ROWS:
    m_ = np.mean(std.d[nm]['전체'], 0); bg = RA[nm]['전체']['배경']
    d4, l5 = np.mean(std.d[nm]['D435'], 0)[0], np.mean(std.d[nm]['L515'], 0)[0]
    OUTJ[nm] = {'rmse': float(m_[0]), 'd105': float(m_[3]), 'bg': bg, 'D435': float(d4), 'L515': float(l5), **SPD[nm]}
    print(f'{nm:20s}{m_[0]:8.2f}{m_[3]:8.2f}{bg:8.2f}{d4:8.2f}{l5:8.2f}{SPD[nm]["fps"]:7.0f}  ({SPD[nm]["how"]})')
json.dump(OUTJ, open(f'{N22}/tta.json', 'w'), indent=1, ensure_ascii=False)
print(f'끝 {time.strftime("%m-%d %H:%M")} | 결과 {N22}')
