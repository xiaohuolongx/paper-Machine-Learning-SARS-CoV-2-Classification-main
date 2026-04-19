clc; clear; close all;

Each_Week = [   '2020_03_14'; '2020_03_21'; '2020_03_28'; '2020_04_04'; ...
                '2020_04_11'; '2020_04_18'; '2020_04_25'; '2020_05_02'; ...
                '2020_05_09'; '2020_05_16'; '2020_05_23'; '2020_05_30'; ...
                '2020_06_06'; '2020_06_13'; '2020_06_20'; '2020_06_27'; ...
                '2020_07_04'; '2020_07_11'; '2020_07_18'; '2020_07_25'; ...
                '2020_08_01'; '2020_08_08'; '2020_08_15'; '2020_08_22'; ...
                '2020_08_29'; '2020_09_05'; '2020_09_12'; '2020_09_19'; ...
                '2020_09_26'; '2020_10_03'; '2020_10_10'; '2020_10_17'; ...
                '2020_10_24'; '2020_10_31'; '2020_11_07'; '2020_11_14'; ...
                '2020_11_21'; '2020_11_28'; '2020_12_05'; '2020_12_12'; ...
                '2020_12_19'; '2020_12_26'; '2021_01_02'; '2021_01_09'; ...
                '2021_01_16'; '2021_01_23'; '2021_01_30'; '2021_02_06'; ...
                '2021_02_13'; '2021_02_20'; '2021_02_27'; '2021_03_06'; ...
                '2021_03_13'; '2021_03_20'; '2021_03_27'; '2021_04_03'; ...
                '2021_04_10'; '2021_04_17'; '2021_04_24'; '2021_05_01'; ...
                '2021_05_08'; '2021_05_15'; '2021_05_22'; '2021_05_29'; ... 
                '2021_06_05'; '2021_06_12'; '2021_06_19'; '2021_06_26'; ...
                '2021_07_03'; '2021_07_10'; '2021_07_17'; '2021_07_24'; ...
                '2021_07_31'; '2021_08_07'; '2021_08_14'; '2021_08_21'; ... 
                '2021_08_28'; '2021_09_04'; '2021_09_11'; '2021_09_18'; ...
                '2021_09_25'; '2021_10_02'; '2021_10_09'; '2021_10_16'; ... 
                '2021_10_23'; '2021_10_30'; '2021_11_06'; '2021_11_13'; ... 
                '2021_11_20'; '2021_11_27'; '2021_12_04'; '2021_12_11'; ...
                '2021_12_18'; '2021_12_25'; '2022_01_01'; '2022_01_08'; ...
                '2022_01_15'; '2022_01_22'; '2022_01_29'; '2022_02_05'; ...
                '2022_02_12'; '2022_02_19'; '2022_02_26'; '2022_03_05'; ...
                '2022_03_12'; '2022_03_19'; '2022_03_26'; '2022_04_02'; ...
                '2022_04_09'; '2022_04_16'; '2022_04_23'; '2022_04_30'; ...
                '2022_05_07'; '2022_05_14'; '2022_05_21'; '2022_05_28'; ...
                '2022_06_04'; '2022_06_11'; '2022_06_18'; '2022_06_25'; ...
                '2022_07_02'; '2022_07_09'; '2022_07_16'; '2022_07_23'; ...
                '2022_07_30'; '2022_08_06'; '2022_08_13'; '2022_08_20'];

          
Alpha = 'B.1.1.7';
Beta = 'B.1.351';
Delta = 'B.1.617.2';
Gamma = 'P.1';
Omicron = 'BA.';

%% load数据

for jj = 1:length(Each_Week)
    freq_seqs = ['H:\新冠数据\mat文件\seqs\','seqs_',Each_Week(jj,:),'.mat'];
    MSA_Seqs = load (freq_seqs);
    freq_header = ['H:\新冠数据\mat文件\header\','header_',Each_Week(jj,:),'.mat'];
    header = load(freq_header);
    header = header';
    freq_tsv = ['H:\新冠数据\tsv文件\',Each_Week(jj,:),'.tsv'];
    delimiterIn   = ' ';
    data_tsv = importdata(freq_tsv, delimiterIn);
    data_cell = regexp(data_tsv, '\t', 'split');%载入数据
    data_header = [];
    data_date = [];
    data_GISAID_clade = [];
 %% 生成所需数据的元胞数组

    for f = 1:length(data_cell)
        data_header = [data_header;data_cell{f}(1,1)];
        if  contains(data_cell{f}(1,5),'/')
            a = data_cell{f}(1,5);
            str = strsplit(a{1,1},'/');
            if str2num(str{1,2}) < 10
            str{1,2} = strcat('0',str{1,2});
            end
            if str2num(str{1,3}) < 10
            str{1,3} = strcat('0',str{1,3});
            end
            str1{1,1} = strcat(str{1,1},'-',str{1,2},'-',str{1,3});
            data_date = [data_date;str1];
        else
            data_date = [data_date;data_cell{f}(1,5)];
        end
        data_GISAID_clade = [data_GISAID_clade;data_cell{f}(1,19)];
    end
    length(data_header)
    data_zhong= [data_header,data_date,data_GISAID_clade];%取头文件和地区
 %% 分出每天数据
    
    data = unique(data_date);%找不同时间
    
    for j = 1:length(data)
       region = [];
       g = 0;
       tf = isequal(data(j,1),{'date'}) ;%剔除空字样
        if tf == 1
            continue
        else
           file = char(data(j,1));
           for k = 1:length(data_zhong)
               if isequal(data_zhong(k,2),data(j,1))
                    g = g+1;
                    region{g,1} = data_zhong(k,1);%找日期对应的头文件
                    region{g,2} = data_zhong(k,2);
                    region{g,3} = data_zhong(k,3);
                else
                    continue
               end
           end
       %% 分类不同毒株数据
 
           
           Alpha1 = [];
           Omicron1 = [];
           B_D_G = [];
           Else1 = [];
           a = 0;
           b = 0;
           c = 0;
           d = 0;
           [m,~]=size(region);
           for i = 1:m
               if contains(region{i,3},Alpha)
                   a = a+1;
                   Alpha1{a,1} = region(i,1);
                   Alpha1{a,2} = region(i,2);
                   Alpha1{a,3} = region(i,3);
               elseif contains(region{i,3},Delta)
                   b = b+1;
                   B_D_G{b,1} = region(i,1);
                   B_D_G{b,2} = region(i,2);
                   B_D_G{b,3} = region(i,3);    
               elseif contains(region{i,3},Omicron)
                   c = c+1;
                   Omicron1{c,1} = region(i,1);
                   Omicron1{c,2} = region(i,2);
                   Omicron1{c,3} = region(i,3); 
               else
                   d = d+1;
                   Else1{d,1} = region(i,1);
                   Else1{d,2} = region(i,2);
                   Else1{d,3} = region(i,3); 
               end
           end
       %% 生成每天不同类的数据
           
           [e,~]=size(Alpha1);
           if e > 150
            row = [];
            for ii = 1 : length(Alpha1)
                id = find(strcmp(header,Alpha1{ii}));
                row = [row ;id];%在header文件找头文件对应的位置行数
            end
            seqs= [];
            seqs = MSA_Seqs(row,:);%在seqs文件找头文件对应的序列
            fre_name = ['H:\新冠数据\天数据\Alpha\',file,'_',Each_Week(jj,:),'.mat'];
            save ( fre_name,'seqs')  % 保存按地区得序列
            
            
            alpha = cell2table(Alpha1);
            A = ['H:\新冠数据\天数据\Alpha\',file,'_',Each_Week(jj,:),'.txt'];
            writetable(alpha,A,'Delimiter',' ')%将分好得头文件和地区信息输出
           end
           
           [e,~]=size(B_D_G);
           if e > 150
            row = [];
            for ii = 1 : length(B_D_G)
                id = find(strcmp(header,B_D_G{ii}));
                row = [row ;id];%在header文件找头文件对应的位置行数
            end
            seqs= [];
            seqs = MSA_Seqs(row,:);%在seqs文件找头文件对应的序列
            fre_name = ['H:\新冠数据\天数据\B_D_G\',file,'_',Each_Week(jj,:),'.mat'];
            save ( fre_name,'seqs')  % 保存按地区得序列
            
            
            b_d_g = cell2table(B_D_G);
            A = ['H:\新冠数据\天数据\B_D_G\',file,'_',Each_Week(jj,:),'.txt'];
            writetable(b_d_g,A,'Delimiter',' ')%将分好得头文件和地区信息输出
           end
           
           [e,~]=size(Omicron1);
           if e > 150
            row = [];
            for ii = 1 : length(Omicron1)
                id = find(strcmp(header,Omicron1{ii}));
                row = [row ;id];%在header文件找头文件对应的位置行数
            end
            seqs= [];
            seqs = MSA_Seqs(row,:);%在seqs文件找头文件对应的序列
            fre_name = ['H:\新冠数据\天数据\Omicron\',file,'_',Each_Week(jj,:),'.mat'];
            save ( fre_name,'seqs')  % 保存按地区得序列
            
            
            omicron = cell2table(Omicron1);
            A = ['H:\新冠数据\天数据\Omicron\',file,'_',Each_Week(jj,:),'.txt'];
            writetable(omicron,A,'Delimiter',' ')%将分好得头文件和地区信息输出
           end
           
           [e,~]=size(Else1);
           if e > 150
            row = [];
            for ii = 1 : length(Else1)
                id = find(strcmp(header,Else1{ii}));
                row = [row ;id];%在header文件找头文件对应的位置行数
            end
            seqs= [];
            seqs = MSA_Seqs(row,:);%在seqs文件找头文件对应的序列
            fre_name = ['H:\新冠数据\天数据\Else\',file,'_',Each_Week(jj,:),'.mat'];
            save ( fre_name,'seqs')  % 保存按地区得序列
            
            
            else1 = cell2table(Else1);
            A = ['H:\新冠数据\天数据\Else\',file,'_',Each_Week(jj,:),'.txt'];
            writetable(else1,A,'Delimiter',' ')%将分好得头文件和地区信息输出
           end
        end
    end
    
end