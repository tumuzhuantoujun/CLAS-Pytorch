import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

def read_seq_info_EDESframe(info_dir):
    '''
    :param info_dir:
    :return: [ED_frame, ES_frame, N_frame]
    '''
    info = (open(info_dir, 'r')).read().split('\n')
    EDESframe = np.zeros((3,), dtype=np.int16)
    for i in range(3):
        EDESframe[i] = int(info[i].split(': ')[-1])
    return EDESframe

def read_A2CA4C_EDESframe(img_dir, pat_num = 450):
    A2C_EDESframe_pat = np.zeros((pat_num, 3), dtype=np.int16)
    A4C_EDESframe_pat = np.zeros((pat_num, 3), dtype=np.int16)
    for i in range(pat_num):
        if i < 9: pat = '000' + str(i + 1)
        elif i < 99: pat = '00' + str(i + 1)
        else: pat = '0' + str(i + 1)
        # Chamber 2
        CH2_dir = img_dir + pat + '/Info_2CH.cfg'
        A2C_EDESframe_pat[i] = read_seq_info_EDESframe(CH2_dir)
        # Chamber 4
        CH4_dir = img_dir + pat + '/Info_4CH.cfg'
        A4C_EDESframe_pat[i] = read_seq_info_EDESframe(CH4_dir)
    return A2C_EDESframe_pat, A4C_EDESframe_pat

def resize_images(imgs, target_shape):
    '''
    :param image sequence with shape: T * H1 * W1
    :param target_shape: (H, W)
    :return: image sequence with shape: T * H * W
    '''
    C = imgs.shape[0]
    H, W = target_shape[0], target_shape[1]
    re_imgs = np.zeros((C, H, W),dtype=np.float32)
    for i in range(C):
        re_imgs[i, :, :] = resize(imgs[i, :, :], output_shape = (H, W), order=1, mode='constant', preserve_range=True, anti_aliasing=True)
    return re_imgs

def read_preprocess_sequences(img_dir, pat_num=450):
    ch2 = np.zeros((pat_num, 10, 256, 256),dtype=np.float32)
    ch4 = np.zeros((pat_num, 10, 256, 256),dtype=np.float32)
    # temporal down sampling (10 frames)
    for i in range(pat_num):
        if i < 9: pat = '000' + str(i + 1)
        elif i < 99: pat = '00' + str(i + 1)
        else: pat = '0' + str(i + 1)
        # Chamber 2
        CH2_dir = img_dir + pat + '/patient' + pat + '_2CH_sequence.mhd'
        CH2_image = sitk.ReadImage(CH2_dir)
        s2 = sitk.GetArrayFromImage(CH2_image)
        s2_ori_cnum = s2.shape[0]
        s2_gap = (s2_ori_cnum - 2) / 8
        select_s2_id = np.zeros(s2_ori_cnum)
        select_s2_id[0] = 1
        select_s2_id[-1] = 1

        if s2_ori_cnum > 10:
            for j in range(8):
                s2_i = np.int(np.round(s2_gap * (j+1)))
                select_s2_id[s2_i] = 1
        else: select_s2_id[:] = 1

        ch2_s = s2[select_s2_id>0,]
        if ch2_ED_ES_inverse_id[pat_id,]==1:
            ch2_s = np.flipud(ch2_s)
        ch2[i,] = resize_images(ch2_s, [256, 256])
        # Chamber 4
        CH4_dir = img_dir + pat + '/patient' + pat + '_4CH_sequence.mhd'
        CH4_image = sitk.ReadImage(CH4_dir)
        s4 = sitk.GetArrayFromImage(CH4_image)
        s4_ori_cnum = s4.shape[0]
        s4_gap = (s4_ori_cnum - 2) / 8
        select_s4_id = np.zeros(s4_ori_cnum)
        select_s4_id[0] = 1
        select_s4_id[-1] = 1
        if s4_ori_cnum > 10:
            for j in range(8):
                s4_i = np.int(np.round(s4_gap * (j + 1)))
                select_s4_id[s4_i] = 1
        else: select_s4_id[:] = 1

        ch4_s = s4[select_s4_id > 0,]
        if ch4_ED_ES_inverse_id[pat_id,]==1:
            ch4_s = np.flipud(ch4_s)
        ch4[i,] = resize_images(ch4_s, [256, 256])
    return ch2, ch4

if __name__ == '__main__':
    A2C_EDESframe, A4C_EDESframe = read_A2CA4C_EDESframe('../data/training/patient')