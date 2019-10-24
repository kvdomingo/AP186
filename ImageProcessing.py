import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.ndimage as img
from PIL import Image, ImageDraw, ImageFilter


class ImageSegment:
    def __init__(self, image):
        if type(image) == str:
            self.image = cv.imread(image)
        elif type(image) == np.ndarray:
            self.image = image
        else:
            raise NotImplementedError
        self.image_hist = np.squeeze(cv.calcHist([self.image.ravel()], [0], None, [256], [0, 255]))
        eps = 1e-4
#         self.image[np.where(-eps < self.image.all() < eps)[0]] = 1e5
        self.cropping = False
        self.sel_rect_endpoint = None
        self.refpt = None
        
    def BGR2NCC(self, img):
        I = img.sum(axis=2)
        b, g, r = cv.split(img)/I
        return I, r, g

    def pixelLikelihood(self, r, mu, sigma):
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(r - mu)**2/(2 * sigma**2))
    
    def ColorPicker(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.refpt = [(x, y)]
            self.cropping = True
            
        elif event == cv.EVENT_LBUTTONUP:
            self.refpt.append((x, y))
            self.cropping = False

            cv.rectangle(self.image, self.refpt[0], self.refpt[1], (0, 255, 0), 2)
            cv.imshow('select ROI', self.image)
            
        elif event == cv.EVENT_MOUSEMOVE and self.cropping:
            self.sel_rect_endpoint = [(x, y)]
            
    def get_ROI(self):
        image = (self.image/self.image.max()).astype('float32')
        clone = image.copy()
        cv.namedWindow('select ROI', cv.WINDOW_NORMAL)
        if image.shape[0] > image.shape[1]:
            cv.resizeWindow('select ROI', 400, 600)
        else:
            cv.resizeWindow('select ROI', 600, 400)
        cv.setMouseCallback('select ROI', self.ColorPicker)

        while True:
            if not self.cropping:
                cv.imshow('select ROI', self.image)
            elif self.cropping and self.sel_rect_endpoint:
                rect_cpy = image.copy()
                cv.rectangle(rect_cpy, self.refpt[0], self.sel_rect_endpoint[0], (0, 255, 0), 1)
                cv.imshow('select ROI', rect_cpy)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('r'):
                image = clone.copy()

            elif key == ord('c'):
                break

        if len(self.refpt) == 2:
            self.roi = clone[self.refpt[0][1]:self.refpt[1][1], self.refpt[0][0]:self.refpt[1][0]]
            cv.imshow("ROI", self.roi)
            cv.waitKey(0)

        cv.destroyAllWindows()
    
    def get_chromaROI(self):
        I, r, g = self.BGR2NCC(self.roi)
        self.mu_r, self.sigma_r = np.mean(r), np.std(r)
        self.mu_g, self.sigma_g = np.mean(g), np.std(g)
        
    def get_chromaIMG(self):
        image = (self.image/self.image.max()).astype('float32')
        I, r, g = self.BGR2NCC(image)
        pr = self.pixelLikelihood(r, self.mu_r, self.sigma_r)
        pg = self.pixelLikelihood(g, self.mu_g, self.sigma_g)
        self.combinedHist = pr * pg
        self.param_out = self.combinedHist.copy()
        
    def get_histROI(self, bins=32, plot_hist=False):
        I, r, g = self.BGR2NCC(self.roi)
        rint = (r*(bins-1)).astype('uint8')
        gint = (g*(bins-1)).astype('uint8')
        rg = np.dstack((rint, gint))
        hist = cv.calcHist([rg], [0, 1], None, [bins, bins], [0, bins-1, 0, bins-1])
        if plot_hist:
            plt.figure(figsize=(5, 5))
            cl_hist = np.clip(hist, 0, bins-1)
            plt.imshow(cl_hist, 'gray', origin='lower')
            plt.xlabel('$g$')
            plt.ylabel('$r$')
            plt.grid(0)
            plt.show()
        self.histROI = hist
        self.bins = bins
        
    def get_histIMG(self):
        bins = self.bins
        I, r, g = self.BGR2NCC(self.image)
        rproj = (r*(bins-1)).astype('uint8')
        gproj = (g*(bins-1)).astype('uint8')
        proj_array = np.zeros(r.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                proj_array[i,j] = self.histROI[rproj[i,j], gproj[i,j]]
        self.combinedHist = proj_array
        self.nparam_out = self.combinedHist.copy()
    
    def plot_segment(self):
        fig = plt.figure(figsize=(16, 9))
        
        ax = fig.add_subplot(121)
        ax.imshow(self.image[:,:,::-1])
        ax.axis('off')
        ax.grid(0)
        
        ax = fig.add_subplot(122)
        ax.imshow(self.combinedHist, 'gray')
        ax.axis('off')
        ax.grid(0)

        plt.show()
        
    def parametric(self, **kwargs):
        self.get_ROI()
        self.get_chromaROI()
        self.get_chromaIMG()
#         self.plot_segment()
        
    def nonparametric(self, **kwargs):
        if self.refpt is None:
            self.get_ROI()
        self.get_histROI(**kwargs)
        self.get_histIMG()
#         self.plot_segment()

    def otsu(self):
        image = self.image.copy()
        image = (image/image.max() * 255).astype('uint8')
        hist = np.squeeze(cv.calcHist([image], [0], None, [256], [0, 255]))
        total = image.size
        top = 256
        sumB = 0
        wB = 0
        maximum = 0.0
        sum1 = np.arange(top) @ hist
        for i in range(top):
            wF = total - wB
            if wB > 0 and wF > 0:
                mF = (sum1 - sumB) / wF
                val = wB * wF * ((sumB/wB) - mF)**2
                if val >= maximum:
                    level = i
                    maximum = val
            wB += hist[i]
            sumB += i * hist[i]
        self.level = level
        
    def main(self, savename=None, **kwargs):
        self.parametric(**kwargs)
        self.nonparametric(**kwargs)
        if self.image.shape[0] > self.image.shape[1]:
            fig = plt.figure(figsize=(16/2, 9/2))
        else:
            fig = plt.figure(figsize=(16/2*2, 9/2))
        
        ax = fig.add_subplot(131)
        ax.imshow(self.image[:,:,::-1])
        ax.axis('off')
        ax.grid(0)
        ax.set_title('original')
        
        ax = fig.add_subplot(132)
        ax.imshow(self.param_out, 'gray')
        ax.axis('off')
        ax.grid(0)
        ax.set_title('parametric')
        
        ax = fig.add_subplot(133)
        ax.imshow(self.nparam_out, 'gray')
        ax.axis('off')
        ax.grid(0)
        ax.set_title('non-parametric')
        
        plt.tight_layout()
        if savename is not None:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.show()
		
		
def find_edge(filename, method):
    flag = None
    shape = Image.open(filename).convert('L')
    
    if method == 'spot':
        shape = shape.filter(ImageFilter.FIND_EDGES)
        flag = 'pil'
    elif method == 'sobel':
        shapex = img.sobel(shape, axis=0)
        shapey = img.sobel(shape, axis=1)
        shape = np.hypot(shapex, shapey)
        flag = 'numpy'
    elif method == 'prewitt':
        shapex = img.prewitt(shape, axis=0)
        shapey = img.prewitt(shape, axis=1)
        shape = np.hypot(shapex, shapey)
        flag = 'numpy'
    elif method == 'laplacian':
        shape = cv.imread(filename, 0)
        shape = cv.GaussianBlur(shape, (3,3), 0)
        shape = cv.Laplacian(shape, cv.CV_64F)
        flag = 'opencv'
    elif method == 'canny':
        shape = cv.imread(filename, 0)
        shape = cv.Canny(shape, 100, 200, 3, L2gradient=True)
        flag = 'opencv'
        
    newname = filename.split('.')[0] + '_' + method + '.png'
    
    if flag == 'pil':
        shape.save(newname)
    elif flag == 'numpy':
        shape = Image.fromarray(shape.astype('uint8'), 'L')
        shape.save(newname)
    elif flag == 'opencv':
        cv.imwrite(newname, shape)
        
    return shape
	
	
def Green(x, y):
    A = 0
    for i in range(1, len(x)):
        A += x[i-1]*y[i] - y[i-1]*x[i]
    return A/2
	
def GreenArea(shape, centroid):
    y, x = np.where(shape > 1)
    x -= centroid[0]
    y -= centroid[1]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    coor = np.array([x, y, r, theta]).T
    coor = coor[coor[:,3].argsort()]
    x, y, r, theta = coor.T
    A = Green(x, y)
    return A
	
	
class WhiteBalance:
    def contrastStretchMod(self, img, p=0.05):
        try:
            img_type = self.datatype
        except AttributeError:
            img_type = str(img.dtype)
        if 'int' in img_type:
            img = img.astype('float64') / np.iinfo(img_type).max
        img_cs = img.copy()
        for i in range(img.shape[2]):
            lo, hi = np.percentile(img.T[i], p), np.percentile(img.T[i], 100-p)
            img_cs.T[i] = (img.T[i] - lo)/(hi - lo) * img.T[i].max() + img.T[i].min()
        img_cs = (img_cs * np.iinfo(img_type).max).astype(img_type)
        return img_cs


    def grayWorld(self, img):
        try:
            img_type = self.datatype
        except AttributeError:
            img_type = str(img.dtype)
        if 'int' in img_type:
            img = img.astype('float64') / np.iinfo(img_type).max
        img_gw = img.copy()
        Bave, Gave, Rave = [img.T[i].mean() for i in range(img.shape[2])]
        Aave = np.mean([Bave, Gave, Rave])
        for i in range(img.shape[2]):
            img_gw.T[i] = img_gw.T[i] * Aave/img.T[i].mean()
        img_gw = (img_gw * np.iinfo(img_type).max).astype(img_type)
        return img_gw
    
    
    def grayWorldMod(self, img):
        try:
            img_type = self.datatype
        except AttributeError:
            img_type = str(img.dtype)
        if 'int' in img_type:
            img = img.astype('float64') / np.iinfo(img_type).max
        img_gw = img.copy()
        Bave, Gave, Rave = [img.T[i].mean() for i in range(img.shape[2])]
        Aave = np.mean([Bave, Gave, Rave])
        for i in range(img.shape[2]):
            img_gw.T[i] = img_gw.T[i] + (Aave - img_gw.T[i].mean())
        img_gw = (img_gw * np.iinfo(img_type).max).astype(img_type)
        return img_gw


    def weightDecision(self, img, n=200):
        img_type = self.datatype
        if 'int' in img_type:
            img = img.astype('float32') / np.iinfo(img_type).max
        channel_std = np.array([img.T[i].std() for i in range(img.shape[2])])
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        x = abs(img.T[2].mean() - img.T[1].mean())
#         weight = n*x/(max(img.T[2].mean(), img.T[1].mean()))
        weight = (abs(img.T[2].mean() - img.T[1].mean()) + abs(max(channel_std) - min(channel_std)))/n
        self.weight = weight
        return weight


    def main(self, img, p=0.05, n=200):
        if type(img) == str:
            img = cv.imread(img)
        elif type(img) == np.ndarray:
            pass
        else:
            raise NotImplementedError
        self.img = img
        self.datatype = str(img.dtype)
        img_hs = self.contrastStretchMod(img, p)
        img_gw = self.grayWorldMod(img)
        w = self.weightDecision(img, n)
        out = w*img_hs + (1 - w)*img_gw
        out = out.astype(self.datatype)
        self.out = out