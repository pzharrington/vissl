import numpy as np

# dr8 to images
# https://github.com/legacysurvey/imagine/blob/master/map/views.py
def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

def _unwise_to_rgb(imgs, bands=[1,2],
                   scale1=1.,
                   scale2=1.,
                   arcsinh=1./20.,
                   mn=-20.,
                   mx=10000., 
                   w1weight=9.):
    img = imgs[0]
    H,W = img.shape

    ## FIXME
    assert(bands == [1,2])
    w1,w2 = imgs
    
    rgb = np.zeros((H, W, 3), np.uint8)

    # Old:
    # scale1 = 50.
    # scale2 = 50.
    # mn,mx = -1.,100.
    # arcsinh = 1.

    img1 = w1 / scale1
    img2 = w2 / scale2

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)

        # intensity -- weight W1 more
        bright = (w1weight * img1 + img2) / (w1weight + 1.)
        I = nlmap(bright)

        # color -- abs here prevents weird effects when, eg, W1>0 and W2<0.
        mean = np.maximum(1e-6, (np.abs(img1)+np.abs(img2))/2.)
        img1 = np.abs(img1)/mean * I
        img2 = np.abs(img2)/mean * I

        mn = nlmap(mn)
        mx = nlmap(mx)

    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)

    rgb[:,:,2] = (np.clip(img1, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,0] = (np.clip(img2, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,1] = rgb[:,:,0]/2 + rgb[:,:,2]/2

    return rgb
