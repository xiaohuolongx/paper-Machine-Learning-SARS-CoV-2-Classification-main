本文件夹中的程序是生成circos图前的预处理程序

都是MATLAB程序，请使用MATLAB运行

首先：fasta_mat.m程序是将GISAID网站下载的数据，经过线下MAFFT比较后的规范数据fasta文件，转换成MATLAB常用的mat文件

第二步：Classification_sequence.m程序是将以周为单位的mat文件，划分成以天为单位，不同毒株的mat文件

第三步：preprocessing.m程序是将以天为单位，不同毒株的mat文件，筛选成DCA过程所需要的MSA

接下来就是进行DCA和circos圈图绘制过程

The program in this folder is a preprocessing program before generating circos diagrams.

They are all MATLAB programs, please use MATLAB to run them.

First of all: the fasta_mat.m program converts the data downloaded from the GISAID website and the standardized data fasta file after offline MAFFT comparison into a mat file commonly used by MATLAB

Step 2: The Classification_sequence.m program divides the mat file in weeks into mat files of different strains in days.

Step 3: The preprocessing.m program is to filter the mat files of different strains into the MSA required for the DCA process based on days.

The next step is to draw the DCA and circos circle diagrams
