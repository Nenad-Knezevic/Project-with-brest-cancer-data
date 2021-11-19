
% CLEAR ALL STUFF FROM WORKSPACE, CLOSING OPEN FIGURES AND CLEAR COMAND
% WINDOW
clear all
close all
clc


% LOADING DATA
tabela =readtable('brest_cancer.csv');

%CONVERT TABLE TO CELL
podaci = table2cell(tabela);
%SELECTING COLUMNS OF INTEREST
vektori_obelezja = podaci(:,3:end);
%CONVERT CELL TO MATRIX
vektori_obelezja = cell2mat(vektori_obelezja);
%VECTOR OF CLASS M-MALIGNANT B-BENIGN
labele = podaci(:,2);   
%VECTOR OF MALIGN
maligni = vektori_obelezja(strcmp(labele,'M'),:);
%VECTOR OF BENIGN
benigni = vektori_obelezja(strcmp(labele,'B'),:);


% EMPYT MATRIX FOR MEAN VALUES
srednjaVrMal = [];
srednjaVrBen = [];


% USING FOR LOOP TO CALCULATE MEAN VALUES
for i=1:size(vektori_obelezja,2)
    srM = mean(maligni(:,i));
    srB = mean(benigni(:,i));
    srednjaVrMal(1,i)=srM;
    srednjaVrBen(1,i)=srB;
    clear srM
    clear srB
end

% EMPTY MATRIX FOR DIFFERENCE BETWEEN MALIGN AND BENIGN MEAN VALUES
raz = [];

% USING FOR LOOP TO CALCULATE DIFFERENCES BETWEEN MALIGN AND BENIGN MEAN
% VALUES
for i=1:size(srednjaVrBen,2)
razTemp = srednjaVrMal(i) - srednjaVrBen(i);
raz(1,i) = razTemp;
clear razTemp
end

% FINDING POSITION OF MAX DIFFERENCE 
disObelezje = max(raz);
[vrsta,kolona] = find(raz==disObelezje);
% FROM TABLE WE SEE THAT VECTOR IS IN POSITION 24  (area_worst)

% FINDING POSITION OF MIN DIFFERENCE
nedisObelezje = min(raz);
[vrstaMin,kolonaMin]=find(raz==nedisObelezje);
% FROM TABLE WE SEE THAT VECTOR IS IN POSITION 12
%(texture_se)


%%%%%% MEAN MEDIAN AND MODE FOR AREA WORST%%%%
dis_srVr_ben = mean(benigni(:,24));
dis_srVr_mal = mean(maligni(:,24));
dis_med_ben = median(benigni(:,24));
dis_med_mal = median(maligni(:,24));
dis_mod_ben = mode(benigni(:,24));
dis_mod_mal = mode(maligni(:,24));

%%%% MEAN MEDIAN AND MODE FOR TEXTURE_SE %%%%%
nedis_srVr_ben = mean(benigni(:,12));
nedis_srVr_mal = mean(maligni(:,12));
nedis_med_ben = median(benigni(:,12));
nedis_med_mal = median(maligni(:,12));
nedis_mod_ben = mode(benigni(:,12));
nedis_mod_mal = mode(maligni(:,12));


% TABLE FOR AREA WORST
dis = table([dis_srVr_ben,dis_srVr_mal,]',[dis_med_ben,dis_med_mal]',[dis_mod_ben,dis_mod_mal]','RowNames',...
    {'Benigni','Maligni'},'VariableNames',{'Mean','Median','Mode'})

% TABLE FOR TEXTURE_SE
nedis = table([nedis_srVr_ben,nedis_srVr_mal]',[nedis_med_ben,nedis_med_mal]',[nedis_mod_ben,nedis_mod_mal]',...
    'RowNames',{'Benigni','Maligni'},'VariableNames',{'Mean','Median','Mode'})

% HISTOGRAMS
figure,histogram(benigni(:,24))
hold on
histogram(maligni(:,24))
title('Diskriminantno obelezje')
xlabel('Vrednost')
ylabel('Kolicina podataka')
legend('Benigni','Maligni')
hold off

figure,histogram(benigni(:,12))
hold on
histogram(maligni(:,12))
title('Nediskriminatorno obelezje')
xlabel('Vrednost')
ylabel('Kolicina podataka')
legend('Benigni','Maligni')
hold off

% KORELATION OF DATA
cor_ben = corr(benigni);
cor_mal = corr(maligni);
figure,scatter(cor_ben(:),cor_mal(:))
hold on
title('Korelacija medju obelezjima')
xlabel('Benigni')
ylabel('Maligni')

% 
%% KNN ALGORITHM

c = cvpartition(labele,'KFold',10);
knn_mdl = fitcknn(vektori_obelezja,labele);
cro_val = crossval(knn_mdl,'CVpartition',c);
pred = kfoldPredict(cro_val);

% RESULTS OF KNN
mat_knn = confusionmat(labele,pred);

TP_knn = mat_knn(1,1); 
FN_knn = mat_knn(1,2); 
FP_knn = mat_knn(2,1); 
TN_knn = mat_knn(2,2);


Osetljivost_knn = TP_knn/(TP_knn+FN_knn);
Specificnost_knn = TN_knn/(FP_knn+TN_knn); 
Tacnost_knn = (TP_knn+TN_knn)/(TP_knn+FN_knn+FP_knn+TN_knn); 
Preciznost_knn = TP_knn/(TP_knn+FP_knn);




%% SVM ALGORITHM

svm_mdl = fitcsvm(vektori_obelezja,labele);
cro_val_svm = crossval(svm_mdl,'CVpartition',c);
pred_svm = kfoldPredict(cro_val_svm);


% RESULTS OF SVM
mat_svm = confusionmat(labele,pred_svm);


TP_svm = mat_svm(1,1); 
FN_svm = mat_svm(1,2); 
FP_svm = mat_svm(2,1); 
TN_svm = mat_svm(2,2);


Osetljivost_svm = TP_svm/(TP_svm+FN_svm); 
Specificnost_svm = TN_svm/(FP_svm+TN_svm); 
Tacnost_svm = (TP_svm+TN_svm)/(TP_svm+FN_svm+FP_svm+TN_svm); 
Preciznost_svm = TP_svm/(TP_svm+FP_svm);




