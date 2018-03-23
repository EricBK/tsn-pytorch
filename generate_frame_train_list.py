import random
import os
import argparse
parser = argparse.ArgumentParser(description="generate frame level file list")
parser.add_argument("--dataset",type=str,default="ucf101")
parser.add_argument("--num_segments",type=int,default=3)
parser.add_argument("--new_length",type=int, default=1)
parser.add_argument("--epochs",type=int,default=80)
args = parser.parse_args()

video_level_list_file = "{}_video_level_list_file.txt".format(args.dataset)
frame_level_list_file = video_level_list_file.replace("video_level","frame_level")
train_data = 'RGB'
bucket_domain = 'http://oy1v528iz.bkt.clouddn.com/'
name_pattern = {'RGB': 'UCF-frames/{video_name}/{idx}.jpg',
                'FLOW':'UCF-frames/{video_name}/image_{idx:05d}.jpg'}
num_segment = args.num_segments
num_length = args.new_length
num_epoch = args.epochs

data_dir = "../data"
def get_label_info(dataset=""):
    label_info = {}
    for child_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir,child_dir)):continue
        if dataset not in child_dir: continue
        for filename in os.listdir(os.path.join(data_dir,child_dir)):
            if "train" not in filename: continue
            with open(os.path.join(data_dir,child_dir,filename),'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.strip().split(" ")
                    video_id = line[0].split("/")[1][:-4]
                    label = line[1]
                    if video_id not in label_info:
                        label_info[video_id] = int(label) - 1
    return label_info

def get_video_frames_from_bucket(bucket="video-data", prefix="UCF-frames/"):
    """
    :param bucked: bucket name
    :param prefix:
    :param des_file:  save info into des_file
    :return:
    """
    temp_file = os.path.join(data_dir,"temp.txt")
    des_file = os.path.join(data_dir,video_level_list_file)
    if os.path.exists(des_file):return
    try:
        print("listing {bucket} info...".format(bucket=bucket))
        if not os.path.exists(temp_file):
            os.system("qshell listbucket {bucket} {prefix} {des_file}".format(bucket=bucket,prefix=prefix,des_file=temp_file))
    except:
        print("command error")
    finally:
        print("finished")
    label_info = get_label_info(args.dataset)
    video_info = {}
    with open(temp_file,'r') as fin,open(des_file,'w') as fout:
        while True:
            line = fin.readline()
            if not line: break
            line = line.split(' ')
            video_id = line[0].split('/')[1]
            if video_id in video_info:
                video_info[video_id] += 1
            else:
                video_info[video_id] = 1
        for video_id in video_info.keys():
            fout.write(bucket_domain+"UCF-frames/{video_id} {num_frames} {label}\n".format(
                video_id=video_id,num_frames=video_info[video_id], label=label_info[video_id]))
def parse_video_level_list(input_file, part=[0.7, 0.3]):
    video_info = []
    with open(input_file, 'r') as fin:
        for line in fin:
            video_path, video_duration, label = line.strip().split(' ')
            if args.dataset == "activitynet":
                video_path = video_path.split('/')[-1] + '.mp4'
            elif args.dataset == "ucf101":
                video_path = video_path.split('/')[-1]
            video_info.append((video_path, int(video_duration), label))
    random.shuffle(video_info)
    video_num = len(video_info)
    return video_info[:int(part[0]*video_num)],video_info[int(part[0]*video_num):]


def generate_frame_info(video_info, shuffle=True):
    random.shuffle(video_info)
    frame_info = []
    for video_path, video_duration, label in video_info:
        avg_duration = video_duration // num_segment
        for i in range(num_segment):
            frame_rng = random.randint(i * avg_duration+1, min((i + 1) * avg_duration - num_length + 1,
                                                             video_duration - 1) - num_length + 1)
            for j in range(num_length):
                if train_data == 'RGB':
                    frame_path = bucket_domain + name_pattern['RGB'].format(video_name=video_path, idx=frame_rng + j)
                    frame_info.append((frame_path, label))
                if train_data == 'FLOW':
                    frame_path_x = bucket_domain + name_pattern['FLOW'].format(video_name=video_path, xy='x',
                                                                               idx=frame_rng + j)
                    frame_path_y = bucket_domain + name_pattern['FLOW'].format(video_name=video_path, xy='y',
                                                                               idx=frame_rng + j)
                    frame_info.append((frame_path_x, label))
                    frame_info.append((frame_path_y, label))

    return frame_info


def main():
    get_video_frames_from_bucket(bucket="video-data",prefix="UCF-frames/")
    v_listfile = os.path.join(data_dir,video_level_list_file)
    video_info_train,video_info_val = parse_video_level_list(v_listfile,part=[0.7,0.3])

    """ video_info_train """
    frame_info_all = []
    for i in range(num_epoch):
        frame_info = generate_frame_info(video_info_train, shuffle=True)
        frame_info_all.extend(frame_info)

    f_listfile = os.path.join(data_dir, frame_level_list_file.replace(".txt","_train.txt"))
    with open(f_listfile, 'w') as fout:
        for item in frame_info_all:
            fout.write('{} {}\n'.format(item[0], item[1]))

    """video_info_val"""
    frame_info = generate_frame_info(video_info_val, shuffle=True)
    f_listfile = os.path.join(data_dir, frame_level_list_file.replace(".txt", "_val.txt"))
    with open(f_listfile, 'w') as fout:
        for item in frame_info:
            fout.write('{} {}\n'.format(item[0], item[1]))

if __name__ == '__main__':
    main()
