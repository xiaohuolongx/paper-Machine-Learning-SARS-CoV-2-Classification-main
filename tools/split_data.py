import os
from shutil import copy, rmtree
import random
from tqdm import tqdm


def main():
    '''
    split_rate  : 测试集划分比例
    init_dataset: 未划分前的数据集路径
    new_dataset : 划分后的数据集路径
    '''

    def makedir(path):
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)

    def safe_copy(src, dst_dir):
        """安全的文件复制函数，处理权限错误"""
        try:
            copy(src, dst_dir)
            return True
        except PermissionError:
            print(f"权限拒绝，跳过文件: {src}")
            return False
        except Exception as e:
            print(f"复制文件时出错 {src}: {e}")
            return False

    split_rate = 0.2
    init_dataset = ''
    new_dataset = ''
    random.seed(0)

    # 过滤掉.git目录和其他非目录文件
    classes_name = [name for name in os.listdir(init_dataset)
                    if os.path.isdir(os.path.join(init_dataset, name))
                    and name != '.git']  # 明确排除.git目录

    print(f"找到的类别: {classes_name}")

    makedir(new_dataset)
    training_set = os.path.join(new_dataset, "train")
    test_set = os.path.join(new_dataset, "test")
    makedir(training_set)
    makedir(test_set)

    for cla in classes_name:
        makedir(os.path.join(training_set, cla))
        makedir(os.path.join(test_set, cla))

    for cla in classes_name:
        class_path = os.path.join(init_dataset, cla)
        # 只获取文件，排除子目录
        img_set = [f for f in os.listdir(class_path)
                   if os.path.isfile(os.path.join(class_path, f))]

        num = len(img_set)
        if num == 0:
            print(f"警告: 类别 '{cla}' 中没有找到图像文件")
            continue

        test_set_index = random.sample(img_set, k=int(num * split_rate))

        print(f"处理类别: {cla}, 总文件数: {num}, 测试集数量: {len(test_set_index)}")

        with tqdm(total=num, desc=f'Class : ' + cla, mininterval=0.3) as pbar:
            copied_count = 0
            skipped_count = 0

            for _, img in enumerate(img_set):
                init_img = os.path.join(class_path, img)

                if img in test_set_index:
                    new_img_dir = os.path.join(test_set, cla)
                else:
                    new_img_dir = os.path.join(training_set, cla)

                # 使用安全的复制函数
                if safe_copy(init_img, new_img_dir):
                    copied_count += 1
                else:
                    skipped_count += 1

                pbar.update(1)

            print(f"类别 {cla} 完成: 成功复制 {copied_count} 个文件, 跳过 {skipped_count} 个文件")
        print()


if __name__ == '__main__':
    main()