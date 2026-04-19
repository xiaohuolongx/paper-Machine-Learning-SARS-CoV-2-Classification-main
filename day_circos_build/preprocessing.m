Filepath = 'H:\新冠数据\毒株分类数据\预处理后\第二版数据集\Delta\';
File = 'H:\新冠数据\天数据\第二版数据集\Else\';
frequency = [90;91;92;93;94;95;96;97;98;99];
files = dir('H:\新冠数据\天数据\第二版数据集\Else\*.mat');
len=length(files);
for k = 1:len
%当前文件夹下创建子文件夹存放处理数据
name = erase(files(k).name,'.mat');
new_folder = sprintf('%s%s',Filepath,name); %指定路径
mkdir(new_folder); %新建文件夹
end
for o = 1:length(frequency)
Survival_Points = zeros(1,len);
    for k = 1: len
    name = erase(files(k).name,'.mat');
    new_folder = sprintf('%s%s',Filepath,name);
    %读入数据
    load([File,name,'.mat']);
    A = seqs;
    %得到列数
    [~,L2]=size(A);
    %得到初始索引
    Major = 0;
    translated_index = zeros(L2,1);
    for i = 1:L2
    Major = Major +1;
    translated_index(i) = Major;
    end
    %给索引标记
    for j = 1:L2
        B=tabulate(A(:,j));
        C=max(B(:,3));
        if C>frequency(o)
           translated_index(j)=0;
        end
    end
    %根据索引删除MSA的列
    i=1:L2;
    id=translated_index(i)==0;
         A(:,id)=[];
    MSA_f = A();
    %保存删除列的MSA
    filename = [new_folder,'\',num2str(frequency(o)),'_filtered_MSA.mat'];
    save(filename,'MSA_f');
    %得到幸存点向量
    i=1:L2;
    id=translated_index(i)==0;
    translated_index(id,:)=[];
    id_loci_remain_original = translated_index.';
    %保存幸存点序列
    filename = [new_folder, '\',num2str(frequency(o)),'_remain_Loci_original.mat'];                                     
    save(filename,'id_loci_remain_original');
    %读入删除列的MSA
    load([new_folder, '\',num2str(frequency(o)),'_filtered_MSA.mat']);
    %删除MSA的重复行
    [MSA_f_uniq, ia, ic]  = unique(MSA_f, 'row', 'stable');
    %保存经过预处理的MSA
    filename = [new_folder,'\',num2str(frequency(o)),'_MSA_f_uniq.mat'];
    save(filename, 'MSA_f_uniq');
    %保存幸存点数量
    Survival_Points(1,k) = length(id_loci_remain_original);
    end
%保存该频率下的幸存点数目向量
filename = [Filepath,'\',num2str(frequency(o)),'_Survival_Points.mat'];
save(filename, 'Survival_Points');
end
for o = 1:length(frequency)
load([Filepath,'\',num2str(frequency(o)),'_Survival_Points.mat'])
Survival_Points_mean(o) = mean(Survival_Points(:));
end
filename = [Filepath,'Survival_Points_mean.mat'];
save(filename, 'Survival_Points_mean');
fid=fopen(['H:\新冠数据\毒株分类数据\预处理后\第二版数据集\Delta\','Survival_Points_mean.txt'],'w');%写入文件路径
for jj=1:length(Survival_Points_mean)
fprintf(fid,'%.4f\r\n',Survival_Points_mean(jj));   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); 
end
fclose(fid);