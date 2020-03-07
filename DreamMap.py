#!/usr/bin/env python

from __future__ import division, print_function  # python 2 compatibility

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, ifftn, fftn
import collections


def kspace_to_image(sig, dim=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.Iterable):
        dim = [dim]

    sig = ifftshift(sig, axes=dim)
    sig = ifftn(sig, axes=dim)
    sig = ifftshift(sig, axes=dim)

    # sig = fftshift(fftn(fftshift(sig, axes=dim), axes=dim), axes=dim)
    # sig = ifftshift(ifftn(ifftshift(sig, axes=dim), axes=dim), axes=dim)

    return sig


def image_to_kspace(sig, dim=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.Iterable):
        dim = [dim]

    sig = fftshift(sig, axes=dim)
    sig = fftn(sig, axes=dim)
    sig = fftshift(sig, axes=dim)

    # sig = ifftshift(ifftn(ifftshift(sig, axes=dim), axes=dim), axes=dim)
    # sig = fftshift(fftn(fftshift(sig, axes=dim), axes=dim), axes=dim)

    return sig


def calc_fa(ste, fid):
    if np.issubdtype(fid.dtype, np.integer):
        ratio = abs(ste) / np.maximum(abs(fid), 1)
    else:  # floating point
        ratio = abs(ste) / np.maximum(abs(fid), np.finfo(fid.dtype).resolution)
    famap = np.rad2deg(np.arctan(np.sqrt(2. * ratio)))
    try:
        famap[famap < 0] = 0.
        famap[np.isnan(famap)] = 0.
        famap[np.isinf(famap)] = 0.
    except:
        pass
    return famap


def approx_sampling(shape, etl, tr=3e-8, dummies=1):

    def genCircularDistance(nz, ny):
        cy = ny // 2
        cz = nz // 2
        y = abs(np.arange(-cy, cy + ny % 2)) / float(cy)
        z = abs(np.arange(-cz, cz + nz % 2)) / float(cz)
        return np.sqrt(y**2 + z[:, np.newaxis]**2)

    ti = genCircularDistance(shape[0], shape[1])**2
    # generate elliptical scanning mask:
    mask = ti > 1
    ti *= etl * tr
    ti += dummies * tr
    ti[mask] = np.nan

    return ti


def DREAM_filter_read(alpha=60., beta=6., tr=3e-3, t1=2., nx=64, etd=1.):
    
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    
    ti = etd * abs(np.linspace(-1, 1, nx, False))**2
    r1s = 1/t1 - np.log(np.cos(beta))/tr
    
    return np.exp(-r1s*ti)[np.newaxis, np.newaxis, :]


def DREAM_filter_fid(alpha=60., beta=6., tr=3e-3, t1=2., ti=None):

    if ti is None:
        ti = tr * np.arange(200)

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)

    # base signal equation for FID
    Sstst = np.sin(beta) * (1 - np.exp(-tr / t1)) / (1 - np.cos(beta) * np.exp(-tr / t1))
    S0 = np.sin(beta) * np.cos(alpha)**2

    r1s = 1/t1 - np.log(np.cos(beta))/tr
    
    # filt = 1/(1 + Sstst/S0*(np.exp(r1s*ti)-1))
    filt = S0/(S0 + Sstst*(np.exp(r1s*ti)-1.))
    filt[np.isnan(ti)] = 0
    
    return filt


def applyFilter(sig, filt, axes=[0, 1], back_transform=True):
    # ifft in lin & par spatial dims
    sig = image_to_kspace(sig, axes)

    while np.ndim(filt) < np.ndim(sig):
        filt = filt[..., np.newaxis]
        
    # multiply with filter
    sig *= filt

    # fft in lin & par spatial dims
    if back_transform:
        sig = kspace_to_image(sig, axes)

    return sig


def local_filter(ste, fid, ti, alpha=60, beta=6, tr=3e-3, t1=2., blur_read=False, fmap=None, nbins=40, niter=2, store_iter = False):

    def iteration(fmap, fid):

        scale = fmap / alpha

        # create nbins
        edges = np.linspace(0., np.nanmax(scale), nbins+1)[1:]
        inds = np.digitize(scale, edges, right=True)
        
        fidnew = np.zeros(fid.shape, dtype=np.complex128)
        for key, item in enumerate(edges):

            # 1) obtain normalized signal evolution
            alpha_ = item * alpha
            beta_ = item * beta
            filt = DREAM_filter_fid(alpha_, beta_, tr, t1, ti)

            # 2) select all voxels in bin
            sig = np.zeros_like(fid)
            sig[inds == key] = fid[inds == key]

            # 3) apply filter for current bin
            # & add bin to solution
            # fidnew += applyFilter(sig, filt, back_transform=False)
            fidnew += applyFilter(sig, filt, back_transform=True)

        # return abs(kspace_to_image(fidnew))
        return abs(fidnew)

    if blur_read and ste.ndim > 2:
        nx = ste.shape[2]
        etd = np.nanmax(ti)
        mean_alpha = calc_fa(ste.mean(), fid.mean())
        mean_beta = mean_alpha / alpha * beta
        filt_ro = DREAM_filter_read(alpha=mean_alpha, beta=mean_beta, tr=tr, t1=t1, nx=nx, etd=etd)
        fid = applyFilter(fid, filt_ro, axes=[2])
        ste = applyFilter(ste, filt_ro, axes=[2])

    if fmap is None:
        fmap = calc_fa(ste, fid)

    if store_iter:
        fmap_iter = np.zeros(np.insert(fmap.shape, fmap.ndim, niter+1), dtype=fmap.dtype)
        print(fmap.shape, fmap_iter.shape)
        fmap_iter[..., 0] = fmap

    for i in range(niter):
        fidnew = iteration(fmap, fid)
        fmap = calc_fa(ste, fidnew)
        if store_iter:
            fmap_iter[..., i+1] = fmap

    if store_iter:
        fmap = fmap_iter

    return fmap, fidnew


def global_filter(ste, fid, ti, alpha=60, beta=6, tr=3e-3, t1=2., blur_read=False):
    nz, ny = ste.shape[:2]
    mean_alpha = calc_fa(ste.mean(), fid.mean())
    mean_beta = mean_alpha / alpha * beta
    filt = DREAM_filter_fid(mean_alpha, mean_beta, tr, t1, ti)
    fid = applyFilter(fid, filt)

    if blur_read and ste.ndim > 2:
        nx = ste.shape[2]
        etd = np.nanmax(ti)
        filt_ro = DREAM_filter_read(alpha=mean_alpha, beta=mean_beta, tr=tr, t1=t1, nx=nx, etd=etd)
        fid = applyFilter(fid, filt_ro, axes=[2])
        ste = applyFilter(ste, filt_ro, axes=[2])

    # filter fid in k-space
    return calc_fa(abs(ste), abs(fid))


def genFieldmaps(fname, meta, fmap_out=None, fcorr_out=None, flocal_out=None, dummies=1, blur_read=False, nbins=40, niter=2, read_dim=2):

    nii = nib.load(fname)
    data = nii.get_data()
    
    if np.ndim(data) != 4:
        print('warning: expected 4D nifti file, found %d dims instead' % np.ndim(data))
        
    if fmap_out is not None:
        fmap = calc_fa(data[..., 0], data[..., 1])
        nii_out = nib.Nifti1Image(fmap, nii.affine, nii.header)
        nii_out.set_data_dtype(np.float32)
        nii_out.to_filename(fmap_out)

    if fcorr_out is None and flocal_out is None:
        return

    if read_dim!=2:
        data = np.moveaxis(data, read_dim, 2)

    if meta['alpha'] is None:
        t1 = meta['t1']
        meta = get_meta_data(args.infile)
        meta['t1'] = t1

    print('alpha = ', meta['alpha'])
    print('beta = ', meta['beta'])
    print('etl = ', meta['etl'])
    print('tr = ', meta['tr'])
    print('t1 = ', meta['t1'])

    nz, ny = data.shape[:2]
    ste = data[..., 0]
    fid = data[..., 1]

    ti = approx_sampling(fid.shape[:2], meta['etl'], meta['tr'], dummies=dummies)
    fmap = global_filter(ste, fid, ti, alpha=meta['alpha'], beta=meta['beta'],
                         tr=meta['tr'], t1=meta['t1'], blur_read=blur_read)

    if fcorr_out is not None:
        nii_out = nib.Nifti1Image(np.moveaxis(fmap, 2, read_dim), nii.affine, nii.header)
        nii_out.set_data_dtype(np.float32)
        nii_out.to_filename(fcorr_out)

    if flocal_out is not None:
        # use fmap from fcorr as initial guess
        fmap, _ = local_filter(ste, fid, ti, alpha=meta['alpha'], beta=meta['beta'],
                               tr=meta['tr'], t1=meta['t1'], blur_read=blur_read,
                               fmap=fmap, nbins=nbins, niter=niter)
        nii_out = nib.Nifti1Image(np.moveaxis(fmap, 2, read_dim), nii.affine, nii.header)
        nii_out.set_data_dtype(np.float32)
        nii_out.to_filename(flocal_out)


def get_meta_data(fname):
    # dcmstack buggy and creates problems, so we need to import it in separate function
    from dcmstack import dcmmeta
    nii_wrp = dcmmeta.NiftiWrapper.from_filename(fname)
    meta = dict()
    meta['alpha'] = nii_wrp.get_meta("CsaSeries.MrPhoenixProtocol.adFlipAngleDegree[0]")
    meta['beta'] = nii_wrp.get_meta("CsaSeries.MrPhoenixProtocol.adFlipAngleDegree[1]")
    meta['etd'] = 1e-3 * \
        nii_wrp.get_meta("CsaSeries.MrPhoenixProtocol.sFastImaging.lEchoTrainDuration")
    meta['tr'] = meta['etd'] / \
        nii_wrp.get_meta("CsaSeries.MrPhoenixProtocol.sFastImaging.lTurboFactor")
    meta['etl'] = round(meta['etd'] / meta['tr'])

    return meta


class NiftiFile(argparse.FileType):
    def __call__(self, string):
        base, ext = os.path.splitext(string)

        if (argparse.FileType()._mode == 'r'):
            file_type = 'input'
        elif (argparse.FileType()._mode == 'w'):
            file_type = 'output'
        else:
            parser.error('NiftiFile: file type not recognized')

        if ext == '':
            string = string + '.nii.gz'  # .nii.gz is default file extension
        else:
            base2, ext2 = os.path.splitext(base)
            if (str.lower(ext) != '.nii') and ((str.lower(ext2) != '.nii') or (str.lower(ext) != '.gz')):
                parser.error('%s file %s should have a .nii or .nii.gz extension' %
                             (file_type, string))

        returnFile = super(NiftiFile, self).__call__(string)
        returnFile.close()
        returnFile = os.path.abspath(returnFile.name)
        return returnFile


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # First, a parser object must be created:
    parser = argparse.ArgumentParser(description='Calculates B1 map from DREAM data\n')

    # Second, we need to add the various command-line options,
    parser.add_argument('-i', '--infile', '--in', type=NiftiFile('r'),
                        help='Input data file', required=True)
    parser.add_argument('-o', '--outfile', '--out', type=NiftiFile('w'),
                        help='Output file for B1 map', required=False)
    parser.add_argument('-g', '--global_out', '--global', type=NiftiFile('w'),
                        help='Output file for globally corrected B1 map', required=False)
    parser.add_argument('-l', '--local_out', '--local', type=NiftiFile('w'),
                        help='Output file for locally corrected B1 map, wip: not working properly', required=False)
    parser.add_argument('--alpha', type=float, default=None, help='DREAM preparation flip angle')
    parser.add_argument('--beta', type=float, default=None, help='DREAM readout flip angle')
    parser.add_argument('--tr', type=float, default=None,
                        help='TR of sequence (excitation to exc.)')
    parser.add_argument('--etl', type=float, default=None,
                        help='Total length of DREAM echo train')
    parser.add_argument('--dummies', type=int, default=1, help='number of dummy scans before DREAM readout')
    parser.add_argument('--t1', type=float, default=2., help='T1 used for correction')
    parser.add_argument('--niter', type=int, default=2,
                        help='number of iterations for local filter correction')
    parser.add_argument('--nbins', type=int, default=40,
                        help='number of bins for local filter correction')    
    parser.add_argument('--blur_read',  action='store_true', dest='blur_read',
                        help='Apply estimated blurring (for average FA) in read for local/global correction')

    parser.add_argument('--read_dim', type=int, default=2, help='position of readout dim. in nifti (count start at 0)')
    args = parser.parse_args()
    meta = dict()
    meta['alpha'] = args.alpha
    meta['beta'] = args.beta
    meta['etl'] = args.etl
    meta['tr'] = args.tr
    meta['t1'] = args.t1

    genFieldmaps(args.infile, meta, fmap_out=args.outfile, fcorr_out=args.global_out,
                 flocal_out=args.local_out, dummies=args.dummies, blur_read=args.blur_read,
                 nbins=args.nbins, niter=args.niter, read_dim=args.read_dim)
