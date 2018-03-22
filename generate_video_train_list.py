import random

video_level_list_file = 'activitynet_1.3_rgb_train_split_1.txt'
frame_level_list_file = 'activitynet_frame_level_video_train_list.txt'
train_data = 'RGB'
bucket_domain = 'http://oxy45khzj.bkt.clouddn.com/'
name_pattern = {'FLOW': 'activitynet-frames-flows/{video_name}/flow_{xy}_{idx:05d}.jpg',
                'RGB': 'activitynet-frames-flows/{video_name}/image_{idx:05d}.jpg'}
num_segment = 3
num_length = 1
num_epoch = 10


def parse_video_level_list(input_file):
    video_info = []
    with open(input_file, 'r') as fin:
        for line in fin:
            video_path, video_duration, label = line.strip().split(' ')
            video_path = video_path[35:] + '.mp4'
            video_info.append((video_path, int(video_duration), label))

    return video_info


def generate_frame_info(video_info, shuffle=True):
    random.shuffle(video_info)
    frame_info = []
    for video_path, video_duration, label in video_info:
        avg_duration = video_duration / num_segment
        for i in xrange(num_segment):
            frame_rng = random.randint(i * avg_duration, min((i + 1) * avg_duration - num_length + 1,
                                                             video_duration - 1) - num_length + 1)
            for j in xrange(num_length):
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
    video_info = parse_video_level_list(video_level_list_file)
    frame_info_all = []
    for i in xrange(num_epoch):
        frame_info = generate_frame_info(video_info, shuffle=True)
        frame_info_all.extend(frame_info)

    with open(frame_level_list_file, 'w') as fout:
        for item in frame_info_all:
            fout.write('{} {}\n'.format(item[0], item[1]))


if __name__ == '__main__':
    main()
