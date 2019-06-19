
# coding: utf-8

# In[ ]:

#
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
#

# In[1]:


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(1)


# In[ ]:


PATH = Path('ILSVRC/')
list(PATH.iterdir())


# In[ ]:


trn_j = json.load(('instances.json').open())
trn_j.keys()


# In[ ]:


# We are saving the key values as constants so that it keeps the code clean and easy to work with
IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']
trn_j[IMAGES][:5]


# In[ ]:


trn_j[ANNOTATIONS][:2]


# In[ ]:


FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'

cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
trn_ids = [o[ID] for o in trn_j[IMAGES]]


# In[ ]:


list((PATH/'Data'/'CLS-LOC'/'train').iterdir())


# In[ ]:


JPEGS = 'Data/CLS-LOC/train'


# In[ ]:


IMG_PATH = PATH/JPEGS
list(IMG_PATH.iterdir())[:5]


# In[ ]:


im0_d = trn_j[IMAGES][0]
im0_d[FILE_NAME],im0_d[ID]


# In[ ]:


trn_anno = collections.defaultdict(lambda:[])
for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
        trn_anno[o[IMG_ID]].append((bb,o[CAT_ID]))

len(trn_anno)


# In[ ]:


im_a = trn_anno[im0_d[ID]]; im_a


# In[ ]:


im0_a = im_a[0]; im0_a


# In[ ]:


cats[7]


# In[ ]:


trn_anno[17]


# In[ ]:


def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])


# In[ ]:


im = open_image(IMG_PATH/im0_d[FILE_NAME])


# In[ ]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# In[ ]:


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# In[ ]:


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


# In[ ]:


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(xy, txt,verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


# In[ ]:


ax = show_img(im)
b = bb_hw(im0_a[0])
draw_rect(ax, b)
draw_text(ax, b[:2], cats[im0_a[1]])


# In[ ]:


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


# In[ ]:


def draw_idx(i):
    im_a = trn_anno[i] # returns binding box values and the object class for image with id i
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)


# In[ ]:


draw_idx(17)


# In[ ]:


def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


# In[ ]:


trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}


# In[ ]:


b,c = trn_lrg_anno[23]
b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[23]), figsize=(5,10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)


# In[ ]:


(PATH/'tmp').mkdir(exist_ok=True)
CSV = PATH/'tmp/lrg.csv'


# In[ ]:


df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids],
    'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn','cat'])
df.to_csv(CSV, index=False)


# In[ ]:


f_model = resnet34
sz=224
bs=64


# In[ ]:


tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms)


# In[ ]:


x,y=next(iter(md.val_dl))


# In[ ]:


show_img(md.val_ds.denorm(to_np(x))[0]);


# In[ ]:


learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam


# In[ ]:


lrf=learn.lr_find(1e-5,100)

# In[ ]:


learn.sched.plot()


# In[ ]:


learn.sched.plot(n_skip=5, n_skip_end=1)


# In[ ]:


lr = 2e-2


# In[ ]:


learn.fit(lr, 1, cycle_len=1)


# In[ ]:


lrs = np.array([lr/1000,lr/100,lr])


# In[ ]:


learn.freeze_to(-2)


# In[ ]:


lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)


# In[ ]:


learn.fit(lrs/5, 1, cycle_len=1)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(lrs/5, 1, cycle_len=2)


# In[ ]:


learn.save('clas_one')


# In[ ]:


learn.load('clas_one')


# In[ ]:


x,y = next(iter(md.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x,preds = to_np(x),to_np(probs)
preds = np.argmax(preds, -1)


# In[ ]:


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.denorm(x)[i]
    b = md.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()
