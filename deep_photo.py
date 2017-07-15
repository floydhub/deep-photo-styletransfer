import argparse
import os
import shutil
import subprocess
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np

def getlaplacian1(i_arr: np.ndarray, consts: np.ndarray, epsilon: float = 0.0000001, win_size: int = 1):
    neb_size = (win_size * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_size * 2 + 1, win_size * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_size:-win_size, win_size:-win_size] + 1).sum() * (neb_size ** 2))

    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(1, win_size * 2 + 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu.T, win_mu) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - win_mu
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')
    row_inds = row_inds.ravel(order='F')
    col_inds = col_inds.ravel(order='F')
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    return (im.astype('float') - min_val) / (max_val - min_val)


def reshape_img(in_img, l=512):
    in_h, in_w, _ = in_img.shape
    if in_h > in_w:
        h2 = l
        w2 = int(in_w * h2 / in_h)
    else:
        w2 = l
        h2 = int(in_h * w2 / in_w)

    return spm.imresize(in_img, (h2, w2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-content_image", help="content image location", default='content.png')
    parser.add_argument("-content_seg", help="content segmentation location", default='')
    parser.add_argument("-style_image", help="style image locations", default='style.png')
    parser.add_argument("-style_blend_weights", help="style image blending weights", default="")
    parser.add_argument("-style_seg", help="style segmentation locations", default='style_seg.png')
    parser.add_argument("-laplacian", help="laplacian file location", default='laplacian.csv')
    parser.add_argument("-output_image", help="output image name", default='out.png')

    parser.add_argument("-image_size", help="Maximum height / width of generated image", default=512)
    parser.add_argument("-gpu", help="GPU indices", default=0)
    parser.add_argument("-multigpu_strategy", help="multi-GPU layer splits", default="")

    parser.add_argument("-content_weight", help="content weight", default=5)
    parser.add_argument("-style_weight", help="style weight", default=10)
    parser.add_argument("-tv_weight", help="tv weight", default=0.001)
    parser.add_argument("-num_iterations", help="iterations", default=2000)
    # parser.add_argument("-normalize_gradients", help="gradient normalisation", action='store_true')
    parser.add_argument("-init", help="initialisation type", default="random", choices=["random", "image"])
    parser.add_argument("-init_image", help="initial image", default="")
    parser.add_argument("-optimizer", help="optimiser", default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("-learning_rate", help="learning rate (adam only)", default=1)
    parser.add_argument("-lbfgs_num_correction", help="lbfgs num correction", default=0)
    parser.add_argument("-print_iter", help="print interval", default=50)
    parser.add_argument("-save_iter", help="save interval", default=100)
    parser.add_argument("-style_scale", help="style scale", default=1.0)
    parser.add_argument("-original_colors", help="use original colours", choices=["0", "1"], default=0)
    parser.add_argument("-pooling", help="pooling type", choices=["max", "avg"], default='max')
    parser.add_argument("-proto_file", help="VGG 19 proto file location", default='models/VGG_ILSVRC_19_layers_deploy.prototxt')
    parser.add_argument("-model_file", help="VGG 19 model file location", default='models/VGG_ILSVRC_19_layers.caffemodel')
    parser.add_argument("-backend", help="backend", choices=["nn", "cudnn", "clnn"], default='cudnn')
    parser.add_argument("-cudnn_autotune", help="cudnn autotune flag", action='store_true')
    parser.add_argument("-seed", help="random number seed", default=-1)
    parser.add_argument("-content_layers", help="VGG 19 content layers", default='relu4_2')
    parser.add_argument("-style_layers", help="VGG 19 style layers", default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
    parser.add_argument("-lambda", help="photorealism weight", dest="photo_lambda", default=1000)
    parser.add_argument("-patch", help="matting patch size", default=3)
    parser.add_argument("-eps", help="matting epsilon", default=1e-7)
    parser.add_argument("-f_radius", help="f radius", default=7)
    parser.add_argument("-f_edge", help="f edge", default=0.05)
    args = parser.parse_args()

    img_size = int(args.image_size)

    if not os.path.exists("/tmp/deep_photo/"):
        os.makedirs("/tmp/deep_photo/")

    img = spi.imread(args.content_image, mode="RGB")
    resized_img = reshape_img(img, img_size)
    content_h, content_w, _ = resized_img.shape
    tmp_content_name=args.content_image.replace(".png",args.image_size+".png")
    spm.imsave(tmp_content_name, resized_img)

    if args.content_seg=="":
        resized_seg_img = resized_img.copy()
        resized_seg_img.fill(0)
        tmp_content_seg_name = tmp_content_name.replace(".png","_seg.png")
    else:
        seg_img = spi.imread(args.content_seg, mode="RGB")
        resized_seg_img = spm.imresize(seg_img, (content_h, content_w))
        tmp_content_seg_name = args.content_seg.replace(".png", args.image_size + ".png")
    spm.imsave(tmp_content_seg_name, resized_seg_img)

    style_images = args.style_image.split(",")

    if args.style_seg!="":
        style_segs = args.style_seg.split(",")
        assert len(style_images)==len(style_segs), '-style_image and -style_seg must have the same number of elements'

    tmp_style_names = []
    tmp_style_seg_names=[]
    for i, style_image in enumerate(style_images):

        style_img = spi.imread(style_image, mode="RGB")
        resized_style_img = reshape_img(style_img, img_size)
        style_h, style_w, _ = resized_style_img.shape
        tmp_style_name = style_image.replace(".png", args.image_size + ".png")
        spm.imsave(tmp_style_name, resized_style_img)
        tmp_style_names.append(tmp_style_name)

        if args.style_seg == "":
            resized_style_seg_img = resized_style_img.copy()
            resized_style_seg_img.fill(0)
            tmp_style_seg_name = tmp_style_name.replace(".png", "_seg.png")
        else:
            style_seg_img = spi.imread(style_segs[i], mode="RGB")
            resized_style_seg_img = spm.imresize(style_seg_img, (style_h, style_w))
            tmp_style_seg_name = style_segs[i].replace(".png", args.image_size + ".png")
        tmp_style_seg_names.append(tmp_style_seg_name)
        spm.imsave(tmp_style_seg_name, resized_style_seg_img)

    if not os.path.exists(args.laplacian):
        print("Calculating matting laplacian for " + str(args.content_image) + " as " + args.laplacian + "...")
        img = im2double(resized_img)
        h, w, c = img.shape
        csr = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-7, 1)
        coo = csr.tocoo()
        zipped = zip(coo.row + 1, coo.col + 1, coo.data)
        with open(args.laplacian, 'w') as out_file:
            out_file.write(str(len(coo.data)) + "\n")
            for row, col, val in zipped:
                out_file.write("%d,%d,%.15f\n" % (row, col, val))

    neural_style_args = ["-content_image", str(tmp_content_name),
                         "-style_image", str(",".join(tmp_style_names)),
                         "-laplacian", str(args.laplacian),
                         "-output_image", str(args.output_image),
                         "-image_size", str(args.image_size),
                         "-gpu", str(args.gpu),
                         "-content_weight", str(args.content_weight),
                         "-style_weight", str(args.style_weight),
                         "-tv_weight", str(args.tv_weight),
                         "-num_iterations", str(args.num_iterations),
                         "-init", str(args.init),
                         "-optimizer", str(args.optimizer),
                         "-learning_rate", str(args.learning_rate),
                         "-lbfgs_num_correction", str(args.lbfgs_num_correction),
                         "-print_iter", str(args.print_iter),
                         "-save_iter", str(args.save_iter),
                         "-style_scale", str(args.style_scale),
                         "-original_colors", str(args.original_colors),
                         "-pooling", str(args.pooling),
                         "-proto_file", str(args.proto_file),
                         "-model_file", str(args.model_file),
                         "-backend", str(args.backend),
                         "-content_layers", str(args.content_layers),
                         "-style_layers", str(args.style_layers),
                         "-lambda", str(args.photo_lambda),
                         "-patch", str(args.patch),
                         "-eps", str(args.eps),
                         "-f_radius", str(args.f_radius),
                         "-f_edge", str(args.f_edge)]

    if args.content_seg!="":
        neural_style_args+=["-content_seg", str(tmp_content_seg_name)]

    if args.style_seg!="":
        neural_style_args+=["-style_seg", str(",".join(tmp_style_seg_names))]

    if args.style_blend_weights!="":
        neural_style_args+=["-style_blend_weights", str(args.style_blend_weights)]

    if args.multigpu_strategy != "":
        neural_style_args+=["-multigpu_strategy", str(args.multigpu_strategy)]

    if args.init_image != "":
        neural_style_args+=["-init_image", str(args.init_image)]

    if args.seed>0:
        neural_style_args+=["-seed", str(args.seed)]

    # if args.normalize_gradients:
    #     neural_style_args +=["-normalize_gradients"]

    if args.cudnn_autotune:
        neural_style_args +=["-cudnn_autotune"]

    cmd = 'th deepmatting_seg.lua ' + " ".join(neural_style_args)

    print("Running "+cmd)
    p = subprocess.Popen("exec bash -c '"+cmd+"'", shell=True)
    p.wait()

    shutil.rmtree("/tmp/deep_photo/", ignore_errors=True)
