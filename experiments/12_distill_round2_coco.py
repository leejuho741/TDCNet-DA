# ============================================================
# 26. 다양한 사진으로 DA 증류 + 더 길게 (25번 이어서). L4 약 4~5시간. 에폭마다 저장 → 끊기면 다시 돌리면 이어서 학습
#   25번 결론: 학생이 TransCG 정확도는 일부 옮겼지만(11.56) 일반화는 못 옮김(ClearGrasp 61.1, 선생 48.6)
#   가설: 특징 따라 하기 손실은 RGB만 필요 → TransCG 밖의 일반 사진(COCO val2017 5천 장)으로도 DA 특징을 따라 하게 하면 일반화가 옮겨질 것
#   KD_SET (결과 폴더 줄 바로 위): 'coco' (본 실험) / 'tcg' (TransCG만, 같은 길이 → '길게'만의 효과를 가르는 대조군. 다른 Colab 세션에서 동시에 돌리면 좋음)
#   1단계: 학생 + head 복사본 학습. 깊이 모델은 원래 s0 그대로(고정) → 학생 특징이 s0와 맞도록 '깊이 출력 따라 하기'는 고정 s0를 통해서
#          TransCG 배치: 특징 + 깊이 출력 따라 하기 + 정답 깊이 + 마스크 / 일반 사진 배치: 특징만
#          train 전체, 6 epoch (25번은 절반 x 6) → 학습량 약 2배, 25번 학생에서 이어서 시작
#   2단계: 학생 고정, proj + head만 TransCG 1 epoch(절반) 조정 (25번처럼). 1단계(proj 원래)와 둘 다 평가
#   평가: TransCG test 1/4 + ClearGrasp real-test (적응 없이). ClearGrasp 사진은 학습에 전혀 안 씀
#   필요: 25번 결과 next25_kd2/student2.tar, next12_scratch/s0, b3_head, transcg_320.zip
#   결과: MyDrive/tdc_finetune/next26_kd3_<KD_SET>/
# ============================================================
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
N13 = f'{TK}/next13_general'; KD_SET = 'coco'   # ← 'coco' = TransCG + 일반 사진(COCO)으로 특징 증류 / 'tcg' = TransCG만 (같은 길이, 대조군)
N22 = f'{TK}/next26_kd3_{KD_SET}'; os.makedirs(N22, exist_ok=True)
sys.stdout = _Tee(sys.stdout.s if isinstance(sys.stdout, _Tee) else sys.stdout, open(f'{N22}/log.txt', 'a'))
print(f'\n######## 26. 다양한 사진으로 DA 증류 + 길게 시작 ({KD_SET}) {time.strftime("%m-%d %H:%M")} | {torch.cuda.get_device_name()}')
B_EPOCHS, B_BS, B_LR = 3, 8, 1e-3
C_STEPS, C_BS, C_LR, W_OUT, W_CONS, W_ANCHOR = 600, 8, 2e-4, 1.0, 0.5, 1.0
torch.manual_seed(0); random.seed(0); np.random.seed(0); rng = random.Random(0)

CG = '/content/cleargrasp'
if not glob.glob(f'{CG}/**/*-transparent-rgb-img.jpg', recursive=True):
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



import torchvision
KD_EPOCHS, KD_BS, LR_S, LR_H = 6, 8, 2e-4, 5e-5
L_FEAT, L_DEP, L_GT, L_MASK, L_EXT = 1.0, 1.0, 0.5, 0.5, 1.0
EVERY = 4
B_STRIDE, LR_P, LR_H2 = 2, 1e-4, 1e-4   # 2단계
HP, WP = DA_SIZE[0] // 14, DA_SIZE[1] // 14


class Student(nn.Module):
    """ResNet-18 (ImageNet) layer1~4 → 층별 어댑터 → DA 3·6·9·12층과 같은 768ch x 18x24 특징 4개 (예전 kd_student와 같은 구조)"""
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool); self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        ad = lambda i: nn.Sequential(nn.Conv2d(i, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.ReLU(True), nn.Conv2d(256, 768, 1))
        self.ad = nn.ModuleList([ad(128 + 256), ad(256 + 512), ad(256 + 512), ad(512)])

    def forward(self, rgb):
        x = (F.interpolate(rgb, size=DA_SIZE, mode='bilinear', align_corners=False) - _MEAN) / _STD
        x = self.stem(x); a = self.l1(x); b = self.l2(a); c = self.l3(b); d = self.l4(c)
        rs = lambda t: F.interpolate(t, size=(HP, WP), mode='bilinear', align_corners=False)
        ins = [torch.cat([rs(b), rs(c)], 1), torch.cat([rs(c), rs(d)], 1), torch.cat([rs(c), rs(d)], 1), rs(d)]
        return [m(t) for m, t in zip(self.ad, ins)]




def loadF(path):
    n_ = FNet(True).to(dev); n_.load_state_dict(torch.load(path, map_location=dev, weights_only=False)['model_state_dict']); return n_.eval()


sc0 = loadF(f'{TK}/next12_scratch/s0/final.tar')
for m_ in (sc0, hb, dab):
    for p_ in m_.parameters(): p_.requires_grad = False

# ---------- 25번 학생 (시작점 + 비교용) ----------
P25 = f'{TK}/next25_kd2/student2.tar'; assert os.path.exists(P25), P25
ck25 = torch.load(P25, map_location=dev, weights_only=False)
stu25 = Student().to(dev).eval(); stu25.load_state_dict(ck25['student']); hS25 = copy.deepcopy(hb).eval(); hS25.load_state_dict(ck25['head'])
dS25 = copy.deepcopy(sc0).eval(); dS25.load_state_dict(ck25['depth'])
stu = Student().to(dev); stu.load_state_dict(ck25['student']); hS = copy.deepcopy(hb); hS.load_state_dict(ck25['head'])

# ---------- 일반 사진: COCO val2017 (5천 장, 라벨 안 씀) ----------
class RGBDS(torch.utils.data.Dataset):
    """RGB만. 무작위 자르기(4:3) + 좌우 뒤집기 + 색 변화 → 320x240"""
    def __init__(self, files): self.f = files
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        im = cv2.imread(self.f[i])
        if im is None: im = np.zeros((480, 640, 3), np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); h, w = im.shape[:2]; r = np.random
        s_ = r.uniform(0.5, 1.0); ch = int(min(h, w * 3 / 4) * s_); cw = int(ch * 4 / 3); y = r.randint(0, h - ch + 1); x = r.randint(0, w - cw + 1)
        im = cv2.resize(im[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
        if r.rand() < 0.5: im = im[:, ::-1]
        if r.rand() < 0.8: im = np.clip((im - 0.5) * r.uniform(0.8, 1.2) + 0.5 + r.uniform(-0.1, 0.1), 0, 1) * r.uniform(0.9, 1.1, 3)
        return torch.from_numpy(np.ascontiguousarray(np.clip(im, 0, 1), dtype=np.float32)).permute(2, 0, 1)


EXT_DL = None
if KD_SET == 'coco':
    CO = '/content/coco'
    if len(glob.glob(f'{CO}/val2017/*.jpg')) < 4900:
        os.makedirs(CO, exist_ok=True); run(f'wget -q -O /content/coco_val.zip http://images.cocodataset.org/zips/val2017.zip && unzip -q -o /content/coco_val.zip -d {CO} && rm /content/coco_val.zip')
    CF = sorted(glob.glob(f'{CO}/val2017/*.jpg')); print('일반 사진 (COCO val2017):', len(CF), '장')
    def ext_iter():
        while True:
            for x in DataLoader(RGBDS(CF), KD_BS, shuffle=True, drop_last=True, num_workers=6, pin_memory=True): yield x
    EXT_DL = ext_iter()


def feat_loss(sf, tf): return sum(((s_ - t_) ** 2).mean() / (t_ ** 2).mean().clamp(min=1e-6) for s_, t_ in zip(sf, tf)) / 4
def teach_feats(rgb):
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP): return [x.float() for x in da_feats(rgb)]


# ---------- 1단계: 학생 + head (깊이 모델은 원래 s0 고정) ----------
CK = f'{N22}/stage1.tar'; start = 0
TR = TCGB('train', True, color=True)
dl = DataLoader(TR, KD_BS, shuffle=True, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)
for p_ in hS.parameters(): p_.requires_grad = True
opt = torch.optim.AdamW([{'params': stu.parameters(), 'lr': LR_S}, {'params': hS.parameters(), 'lr': LR_H}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_S, LR_H], total_steps=KD_EPOCHS * len(dl), pct_start=0.03)
if os.path.exists(CK):
    c_ = torch.load(CK, map_location=dev, weights_only=False); stu.load_state_dict(c_['student']); hS.load_state_dict(c_['head'])
    opt.load_state_dict(c_['opt']); sched.load_state_dict(c_['sched']); start = c_['epoch']; print(f'1단계 이어서: {start}/{KD_EPOCHS} epoch 끝난 상태')
pwt = torch.tensor(10., device=dev); torch.manual_seed(start)
print(f'1단계: train {len(TR)}장 x {KD_EPOCHS} epoch, batch {KD_BS} ({len(dl)} step/epoch), 일반 사진 {"사용" if EXT_DL else "안 씀"}')
for ep in range(start, KD_EPOCHS):
    stu.train(); hS.train(); t0 = time.time(); S = np.zeros(5); n = 0
    for b in dl:
        b = to_dev(b); rgb, depth, gt, gm, cams = b['rgb'], b['depth'], b['gt'], b['mask'], b['cam']
        tf = teach_feats(rgb)
        with torch.no_grad(): tp = torch.sigmoid(hb(rgb, depth, tf)).float(); tD = sc0(rgb, depth, tf).float().view_as(depth)
        with torch.autocast('cuda', dtype=torch.bfloat16): sf = stu(rgb)
        sf = [x.float() for x in sf]
        l_feat = feat_loss(sf, tf)
        D = sc0(rgb, depth, sf).float().view_as(depth)                    # 고정 s0를 통과 → 학생 특징이 s0와 맞도록
        l_dep = (D - tD).abs().mean() + 4 * ((D - tD).abs()[gm].mean() if gm.any() else D.sum() * 0)
        l_gt = tdc_loss(D, gt, gm, cams)[0]
        logit = hS(rgb, depth, sf); p = torch.sigmoid(logit); gmf = gm.float()
        dice = 1 - (2 * (p * gmf).sum((1, 2)) + 1) / (p.sum((1, 2)) + gmf.sum((1, 2)) + 1)
        l_mask = F.binary_cross_entropy_with_logits(logit, tp) + F.binary_cross_entropy_with_logits(logit, gmf, pos_weight=pwt) + dice.mean()
        loss = L_FEAT * l_feat + L_DEP * l_dep + L_GT * l_gt + L_MASK * l_mask
        l_ext = torch.zeros((), device=dev)
        if EXT_DL is not None:                                             # 일반 사진: 특징 따라 하기만
            xe = next(EXT_DL).to(dev, non_blocking=True); te = teach_feats(xe)
            with torch.autocast('cuda', dtype=torch.bfloat16): se = stu(xe)
            l_ext = feat_loss([x.float() for x in se], te); loss = loss + L_EXT * l_ext
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(stu.parameters()) + list(hS.parameters()), 1.0); opt.step(); sched.step()
        S += [float(l_feat), float(l_ext), float(l_dep) * 1000, float(l_gt) * 1000, float(l_mask)]; n += 1
        if n % 1000 == 0: print(f'    {n}/{len(dl)} 특징 TransCG {S[0] / n:.4f} 일반 {S[1] / n:.4f} 깊이따라하기 {S[2] / n:.2f} 정답 {S[3] / n:.2f} 마스크 {S[4] / n:.3f} | {(time.time() - t0) / 60:.0f}분')
    print(f'  epoch {ep + 1}/{KD_EPOCHS} | 특징 TransCG {S[0] / n:.4f} 일반 {S[1] / n:.4f} 깊이따라하기 {S[2] / n:.2f} 정답 {S[3] / n:.2f} 마스크 {S[4] / n:.3f} | {(time.time() - t0) / 60:.1f}분')
    torch.save({'student': stu.state_dict(), 'head': hS.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep + 1}, CK)
stu.eval(); hS.eval()
for p_ in stu.parameters(): p_.requires_grad = False

# ---------- 2단계: 학생 고정, proj + head만 TransCG 절반 1 epoch ----------
CK2 = f'{N22}/stage2.tar'; dS = copy.deepcopy(sc0); hS2 = copy.deepcopy(hS)
if os.path.exists(CK2):
    c_ = torch.load(CK2, map_location=dev, weights_only=False); dS.load_state_dict(c_['depth']); hS2.load_state_dict(c_['head']); print('2단계 불러옴')
else:
    freeze_base(dS)
    for p_ in hS2.parameters(): p_.requires_grad = True
    TR2 = TCGB('train', True, color=True); TR2.s = TR2.s[::B_STRIDE]
    dl2 = DataLoader(TR2, KD_BS, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    opt2 = torch.optim.AdamW([{'params': dS.proj.parameters(), 'lr': LR_P}, {'params': hS2.parameters(), 'lr': LR_H2}], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=[LR_P, LR_H2], total_steps=len(dl2), pct_start=0.05); t0 = time.time(); S = np.zeros(2); n = 0
    for b in dl2:
        b = to_dev(b); rgb, depth, gt, gm, cams = b['rgb'], b['depth'], b['gt'], b['mask'], b['cam']
        tf = teach_feats(rgb)
        with torch.no_grad():
            tp = torch.sigmoid(hb(rgb, depth, tf)).float(); tD = sc0(rgb, depth, tf).float().view_as(depth); sf = [x.float() for x in stu(rgb)]
        D = dS(rgb, depth, sf).float().view_as(depth)
        l_dep = (D - tD).abs().mean() + 4 * ((D - tD).abs()[gm].mean() if gm.any() else D.sum() * 0); l_gt = tdc_loss(D, gt, gm, cams)[0]
        logit = hS2(rgb, depth, sf); l_mask = F.binary_cross_entropy_with_logits(logit, tp) + F.binary_cross_entropy_with_logits(logit, gm.float(), pos_weight=pwt)
        loss = L_DEP * l_dep + L_GT * l_gt + L_MASK * l_mask
        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); sch2.step(); S += [float(l_dep) * 1000, float(l_gt) * 1000]; n += 1
    print(f'  2단계 끝 | 깊이따라하기 {S[0] / n:.2f} 정답 {S[1] / n:.2f} | {(time.time() - t0) / 60:.1f}분')
    torch.save({'depth': dS.state_dict(), 'head': hS2.state_dict()}, CK2)
dS.eval(); hS2.eval()


def pipe(feats_fn, depth_net, head, rgb, depth):
    f = feats_fn(rgb); m = ero(hyst_gpu(torch.sigmoid(head(rgb, depth, f)).float()), 2)
    return gate(depth_net(rgb, depth, f).float().view_as(depth), depth, m), m


sF = lambda net: (lambda rgb: [x.float() for x in net(rgb)])
V = 'COCO+TransCG' if KD_SET == 'coco' else 'TransCG만'
def outs(rgb, depth):
    O = {}
    O['선생: s0 + DA (현재 최종)'] = pipe(da_feats, sc0, hb, rgb, depth)
    T = tdc0(rgb, None, depth).float().view_as(depth); mT = O['선생: s0 + DA (현재 최종)'][1]
    O['기준: TDCNet + DA head 게이트'] = (gate(T, depth, mT), mT)
    O['25번 학생 ★ (proj 조정)'] = pipe(sF(stu25), dS25, hS25, rgb, depth)
    O['25번 학생 · 원래 s0'] = pipe(sF(stu25), sc0, hS25, rgb, depth)
    O[f'26 학생 ({V}) · 원래 s0'] = pipe(sF(stu), sc0, hS, rgb, depth)
    O[f'26 학생 ({V}) · proj 조정'] = pipe(sF(stu), dS, hS2, rgb, depth)
    return O


SUB = list(range(0, len(TEST), EVERY)); acc = Acc(); std = StdAcc(); t0 = time.time(); i0 = 0
with torch.no_grad():
    for bi, b in enumerate(DataLoader(Subset(TEST, SUB), 32, shuffle=False, num_workers=8, pin_memory=True)):
        b = to_dev(b); rgb, depth, gt, gm = b['rgb'], b['depth'], b['gt'], b['mask']; cams = b['cam'].tolist(); i0 += len(cams); reg = regions(gm)
        for nm, (o, m) in outs(rgb, depth).items(): score(acc, nm, o, gt, gm, reg, cams, mask_in=m); std.add(nm, o, gt, gm, cams, {1: 'D435', 2: 'L515'})
        if bi % 50 == 0: print(f'  [TransCG] {i0}/{len(SUB)} {(time.time() - t0) / 60:.1f}분')
R = report(acc, f'26. 증류 {V} (TransCG test 1/{EVERY})', f'{N22}/transcg_regions.json')
RS = std.table(f'26. TransCG test 1/{EVERY} 표준 지표', list(acc.d), ['D435', 'L515'])
std2 = StdAcc()
with torch.no_grad():
    for b in DataLoader(CGDS(TST), 16, shuffle=False, num_workers=8):
        b = to_dev(b)
        for nm, (o, m) in outs(b['rgb'], b['depth']).items(): std2.add(nm, o, b['gt'], b['mask'], b['cam'].tolist(), {1: 'D435', 2: 'D415'})
RC = std2.table(f'26. ClearGrasp real-test {len(TST)}장 (적응 없이)', list(acc.d), ['D435', 'D415'])
# 특징이 DA와 얼마나 닮았나 (ClearGrasp 사진에서, 상대 MSE, 낮을수록 닮음)
with torch.no_grad():
    fl_ = {k: [] for k in ('25번 학생', f'26 학생 ({V})')}
    for b in DataLoader(CGDS(TST), 16, shuffle=False, num_workers=8):
        r_ = b['rgb'].to(dev); tf = teach_feats(r_)
        fl_['25번 학생'].append(float(feat_loss(sF(stu25)(r_), tf))); fl_[f'26 학생 ({V})'].append(float(feat_loss(sF(stu)(r_), tf)))
print('\nClearGrasp 사진에서 DA 특징과의 차이 (상대 MSE):', {k: round(float(np.mean(v)), 4) for k, v in fl_.items()})

g = lambda n, k, c='전체': R[n][c][k]
print(f'\n\n################ 26. 요약 ({V}, TransCG test 1/{EVERY}, mm) ################')
print(f'{"":40s}{"투명":>8s}{"δ1.05":>8s}{"경계":>8s}{"주변":>8s}{"배경":>8s}{"D435":>8s}{"L515":>8s}{"IoU":>7s}{"CG실제":>8s}')
for nm in acc.d:
    cg = RC.get(nm, {}).get('전체', [float('nan')])[0]
    print(f'{nm[:40]:40s}' + ''.join(f'{g(nm, k):8.2f}' for k in ['투명 RMSE (256x144)', '투명 δ1.05 (256x144)', '경계', '주변', '배경'])
          + f'{g(nm, "투명 RMSE (256x144)", "D435"):8.2f}{g(nm, "투명 RMSE (256x144)", "L515"):8.2f}{g(nm, "마스크 IoU"):7.3f}{cg:8.2f}')
json.dump({'transcg': RS, 'cleargrasp': RC, 'feat_gap_cg': {k: float(np.mean(v)) for k, v in fl_.items()}}, open(f'{N22}/kd3.json', 'w'), indent=1, ensure_ascii=False)
print(f'끝 {time.strftime("%m-%d %H:%M")} | 결과 {N22}')
