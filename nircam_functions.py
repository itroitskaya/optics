import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax.tree_util import tree_map

import dLux as dl
import jax.numpy as np
from jax import custom_jvp, pure_callback, vmap, grad, jit

import webbpsf
import poppy
import pysiaf
import scipy
from astropy.io import fits
import astropy.units as u
from tqdm.notebook import tqdm
from dLux.optics import OpticalLayer
from dLux.detectors import DetectorLayer

import jax.numpy as jnp
import scipy.special
from jax import custom_jvp, pure_callback, vmap


# see https://github.com/google/jax/issues/11002

def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: -cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv

jv_callback = generate_bessel(lambda v, x: scipy.special.j1(x) if v == 1 else scipy.special.jv(v, x))




#working Bessel J1

SQ2OPI =  7.9788456080286535587989E-1   
THPIO4 =  2.35619449019234492885        

RP = jnp.array([
-8.99971225705559398224E8, 4.52228297998194034323E11,
-7.27494245221818276015E13, 3.68295732863852883286E15,])
RQ = jnp.array([
 1.0, 6.20836478118054335476E2, 2.56987256757748830383E5, 8.35146791431949253037E7, 
 2.21511595479792499675E10, 4.74914122079991414898E12, 7.84369607876235854894E14, 
 8.95222336184627338078E16, 5.32278620332680085395E18,])

PP = jnp.array([
 7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0, 
 5.11207951146807644818E0, 8.42404590141772420927E0, 5.21451598682361504063E0, 1.00000000000000000254E0,])
PQ = jnp.array([
 5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0, 
 5.07386386128601488557E0, 8.39985554327604159757E0, 5.20982848682361821619E0, 9.99999999999999997461E-1,])

QP = jnp.array([
 5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1, 
 3.66779609360150777800E2, 7.10856304998926107277E2, 5.97489612400613639965E2, 2.11688757100572135698E2, 2.52070205858023719784E1,])
QQ  = jnp.array([
 1.0, 7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3, 
 9.56231892404756170795E3, 7.99704160447350683650E3, 2.82619278517639096600E3, 3.36093607810698293419E2,])

YP = jnp.array([
 1.26320474790178026440E9,-6.47355876379160291031E11, 1.14509511541823727583E14,
 -8.12770255501325109621E15, 2.02439475713594898196E17,-7.78877196265950026825E17,])
YQ = jnp.array([
 5.94301592346128195359E2, 2.35564092943068577943E5, 7.34811944459721705660E7, 
 1.87601316108706159478E10, 3.88231277496238566008E12, 6.20557727146953693363E14, 
 6.87141087355300489866E16, 3.97270608116560655612E18,])

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1

def j1_small(x):
    z = x * x
    w = jnp.polyval(RP, z) / jnp.polyval(RQ, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w

def j1_large_c(x):    
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP, z) / jnp.polyval(PQ, z)
    q = jnp.polyval(QP, z) / jnp.polyval(QQ, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j1(x):
    """
    Bessel function of order one - using the implementation from CEPHES, translated to Jax.
    """
    return jnp.sign(x)*jnp.where(jnp.abs(x) < 5.0, j1_small(jnp.abs(x)),j1_large_c(jnp.abs(x)))





def get_pixel_positions(npixels, pixel_scales = None, indexing = 'xy'):
    # Turn inputs into tuples
    if isinstance(npixels, int):
        npixels = (npixels,)

        if pixel_scales is None:
            pixel_scales = (1.,)
        elif not isinstance(pixel_scales, (float, Array)):
            raise ValueError("pixel_scales must be a float or Array if npixels "
                             "is an int.")
        else:
            pixel_scales = (pixel_scales,)
        
    # Check input 
    else:
        if pixel_scales is None:
            pixel_scales = tuple([1.]*len(npixels))
        elif isinstance(pixel_scales, float):
            pixel_scales = tuple([pixel_scales]*len(npixels))
        elif not isinstance(pixel_scales, tuple):
            raise ValueError("pixel_scales must be a tuple if npixels is a tuple.")
        else:
            if len(pixel_scales) != len(npixels):
                raise ValueError("pixel_scales must have the same length as npixels.")
    
    def pixel_fn(n, scale):
        pix = np.arange(n) - n / 2.
        pix *= scale
        return pix
    
    pixels = tree_map(pixel_fn, npixels, pixel_scales)

    positions = np.array(np.meshgrid(*pixels, indexing=indexing))

    return np.squeeze(positions)

class InvertY(OpticalLayer):
    def __init__(self):
        super().__init__("InvertY")

    def __call__(self, wavefront):
        return wavefront.invert_y()
    
class InvertX(OpticalLayer):
    def __init__(self):
        super().__init__("InvertX")

    def __call__(self, wavefront):
        return wavefront.invert_x()
    
class InvertXY(OpticalLayer):
    def __init__(self):
        super().__init__("InvertXY")

    def __call__(self, wavefront):
        return wavefront.invert_x_and_y()
    
class Pad(OpticalLayer):
    npix_out: int  

    def __init__(self, npix_out):
        self.npix_out = int(npix_out)
        super().__init__("Pad")
    
    def __call__(self, wavefront):
        return wavefront.pad_to(self.npix_out)
    
class Crop(OpticalLayer):
    npix_out: int   

    def __init__(self, npix_out):
        self.npix_out = int(npix_out)
        super().__init__("Crop")
    
    def __call__(self, wavefront):
        # Get relevant parameters
        return wavefront.crop_to(self.npix_out)
    
class NircamCirc(OpticalLayer):
    sigma: float
    diam: None
    npix: None
    oversample: None
    
    def __init__(self, sigma, diam, npix, oversample):
        self.sigma = float(sigma)
        self.diam = float(diam)
        self.npix = int(npix)
        self.oversample = int(oversample)
        
        super().__init__("NircamCirc")
    
    def __call__(self, wavefront):
        
        #jax.debug.print("wavelength: {}", vars(wavefront))
        
        return wavefront.multiply_amplitude(self.get_transmission(wavefront.wavelength, wavefront.pixel_scale))
    
    def get_transmission(self, wavelength, pixelscale):
        #s = 2035
        #jax.debug.print("wavelength start: {}", wavelength)
        #jax.debug.print("pixelscale rad: {}", pixelscale)
        
        pixelscale = pixelscale * 648000.0 / np.pi #from rad to arcsec 
        
        #pixelscale = wavelength * 648000.0 / np.pi
        #pixelscale = pixelscale / self.diam / self.oversample
        
        #jax.debug.print("pixelscale arcsec: {}", pixelscale)
        
        npix = self.npix * self.oversample
        
        x, y = get_pixel_positions((npix, npix), (pixelscale, pixelscale))
        
        #jax.debug.print("x: {}", x[s:-s,s:-s][0].tolist())
        
        r = np.sqrt(x ** 2 + y ** 2)
        #jax.debug.print("r: {}", r[s:-s,s:-s][0].tolist())
        
        sigmar = self.sigma * r
        
        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        bessel_j1_zero2 = scipy.special.jn_zeros(1, 2)[1]
        
        sigmar = sigmar.clip(np.finfo(sigmar.dtype).tiny, bessel_j1_zero2)  # avoid divide by zero -> NaNs
        
        #transmission = (1 - (2 * scipy.special.j1(sigmar) / sigmar) ** 2)
        #transmission = (1 - (2 * jax.scipy.special.bessel_jn(sigmar,v=1)[1] / sigmar) ** 2)
        #transmission = (1 - (2 * jv(1, sigmar) / sigmar) ** 2)
        transmission = (1 - (2 * j1(sigmar) / sigmar) ** 2)
        
        #jax.debug.print("transmission: {}", transmission[s:-s,s:-s][0].tolist())
        #jax.debug.print("transmission: {}", transmission[s:-s,s:-s].tolist())
        
        transmission = np.where(r == 0, 0, transmission)

        # the others have two, one in each corner, both halfway out of the 10x10 box.
        transmission = np.where(
            ((y < -5) & (y > -10)) &
            (np.abs(x) > 7.5) &
            (np.abs(x) < 12.5),
            np.sqrt(1e-3), transmission
        )
        transmission = np.where(
            ((y < 10) & (y > 8)) &
            (np.abs(x) > 9) &
            (np.abs(x) < 11),
            np.sqrt(1e-3), transmission
        )
        
        # mask holder edge
        transmission = np.where(y > 10, 0.0, transmission)

        # edge of mask itself
        # TODO the mask edge is complex and partially opaque based on CV3 images?
        # edge of glass plate rather than opaque mask I believe. To do later.
        # The following is just a temporary placeholder with no quantitative accuracy.
        # but this is outside the coronagraph FOV so that's fine - this only would matter in
        # modeling atypical/nonstandard calibration exposures.
        transmission = np.where((y < -11.5) & (y > -13), 0.7, transmission)
        
        return transmission
    

class NIRCamFieldAndWavelengthDependentAberration(OpticalLayer):
    opd: None
        
    zernike_coeffs: None
    defocus_zern: None
    tilt_zern: None
        
    focusmodel: None
    deltafocus: None
    opd_ref_focus: None
        
    ctilt_model: None
    tilt_offset: None
    tilt_ref_offset: None
    
    def __init__(self, instrument, opd, zernike_coeffs):
        super().__init__("NIRCamFieldAndWavelengthDependentAberration")
        
        self.opd = np.asarray(opd, dtype=float)
        self.zernike_coeffs = np.asarray(zernike_coeffs, dtype=float)

        # Polynomial equations fit to defocus model. Wavelength-dependent focus 
        # results should correspond to Zernike coefficients in meters.
        # Fits were performed to the SW and LW optical design focus model 
        # as provided by Randal Telfer. 
        # See plot at https://github.com/spacetelescope/webbpsf/issues/179
        # The relative wavelength dependence of these focus models are very
        # similar for coronagraphic mode in the Zemax optical prescription,
        # so we opt to use the same focus model in both imaging and coronagraphy.
        defocus_to_rmswfe = -1.09746e7 # convert from mm defocus to meters (WFE)
        sw_focus_cf = np.array([-5.169185169, 50.62919436, -201.5444129, 415.9031962,  
                                -465.9818413, 265.843112, -59.64330811]) / defocus_to_rmswfe
        lw_focus_cf = np.array([0.175718713, -1.100964635, 0.986462016, 1.641692934]) / defocus_to_rmswfe

        # Coronagraphic tilt (`ctilt`) offset model
        # Primarily effects the LW channel (approximately a 0.031mm diff from 3.5um to 5.0um).
        # SW module is small compared to LW, but we include it for completeness.
        # Values have been determined using the Zernike offsets as reported in the 
        # NIRCam Zemax models. The center reference positions will correspond to the 
        # NIRCam target acquisition filters (3.35um for LW and 2.1um for SW)
        sw_ctilt_cf = np.array([125.849834, -289.018704]) / 1e9
        lw_ctilt_cf = np.array([146.827501, -2000.965222, 8385.546158, -11101.658322]) / 1e9

        # Get the representation of focus in the same Zernike basis as used for
        # making the OPD. While it looks like this does more work here than needed
        # by making a whole basis set, in fact because of caching behind the scenes
        # this is actually quick
        basis = poppy.zernike.zernike_basis_faster(
            nterms=len(self.zernike_coeffs),
            npix=self.opd.shape[0],
            outside=0
        )
        self.defocus_zern = np.asarray(basis[3])
        self.tilt_zern = np.asarray(basis[2])

        # Which wavelength was used to generate the OPD map we have already
        # created from zernikes?
        if instrument.channel.upper() == 'SHORT':
            self.focusmodel = sw_focus_cf
            opd_ref_wave = 2.12
        else:
            self.focusmodel = lw_focus_cf
            opd_ref_wave = 3.23
        
        self.opd_ref_focus = np.polyval(self.focusmodel, opd_ref_wave)
        
        print("opd_ref_focus: {}", self.opd_ref_focus)

        # If F323N or F212N, then no focus offset necessary
        if ('F323N' in instrument.filter) or ('F212N' in instrument.filter):
            self.deltafocus = lambda wl: 0
        else:
            self.deltafocus = lambda wl: np.polyval(self.focusmodel, wl) - self.opd_ref_focus

        # Apply wavelength-dependent tilt offset for coronagraphy
        # We want the reference wavelength to be that of the target acq filter
        # Final offset will position TA ref wave at the OPD ref wave location
        #   (wave_um - opd_ref_wave) - (ta_ref_wave - opd_ref_wave) = wave_um - ta_ref_wave
        
        if instrument.channel.upper() == 'SHORT':
            self.ctilt_model = sw_ctilt_cf
            ta_ref_wave = 2.10
        else: 
            self.ctilt_model = lw_ctilt_cf
            ta_ref_wave = 3.35
            
        self.tilt_ref_offset = np.polyval(self.ctilt_model, ta_ref_wave)
        
        print("tilt_ref_offset: {}", self.tilt_ref_offset)

        self.tilt_offset = lambda wl: np.polyval(self.ctilt_model, wl) - self.tilt_ref_offset
        
    def __call__(self, wavefront):
        
        wavelength = wavefront.wavelength * 1e6
        
        #jax.debug.print("wavelength: {}", wavelength)
        
        mod_opd = self.opd - self.deltafocus(wavelength) * self.defocus_zern
        mod_opd = mod_opd + self.tilt_offset(wavelength) * self.tilt_zern
        
        return wavefront.add_opd(mod_opd)


