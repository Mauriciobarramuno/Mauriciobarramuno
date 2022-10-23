%% Predictive model of university dropout V1.2022.
% Predictive model of university dropout in students of different programs based on machine learning. 
% This model is based on the RUSboosted trees classifier. 
% The sample included 946 students belonging to seven different knowledge areas of a university in Chile. 
% This script was build in matlab R2106a.

%% How to use the classifier.
% Please build a New_Test_Matrix that includes the following ten variables in the same order:
% 1) Age = Age in years completed.
% 2) Mother’s education level = (1) Unknow, (2) up to complete high school, (3) up to technical degree, (4) up to complete university degree, (5) postgraduate, (6) military academic training.
% 3) Father’s education level = (1) Unknow, (2) up to complete high school, (3) up to technical degree, (4) up to complete university degree, (5) postgraduate, (6) military academic training.
% 4) Gender = Male (1) - Female (0)
% 5) School origin = (1) Municipal. (2) Subsidized. (3) Private.
% 6) Years to enter university = Number of years completed between high school and entry to university
% 7) Gratuity = Yes (1) – No (0).
% 8) Number of subjects failed. = Number of subjects failed in last year.
% 9) Grade point average = Grade between 1.0 (lowest) and 7.0 (highest)
% 10) Number of subjects enrolled per year = Number of subjects taken in the year
% 
% yfit = trainedClassifier.predictFcn(New_Test_Matrix);
% Each row of "Yfit" represents the result of the classifier for each student. 0 dropout, 1 non-dropout. 

%% Script info
% Script has been published by Mauricio Barramuño Medina, Autonomous University of Chile. 
% Please contact me mauricio.barramuno@uautonoma.cl if any issues or message me on 
% https://www.researchgate.net/profile/Mauricio-Barramuno-Medina

%% Code
clear;clc;close all

load('Model.mat');
load('Example.mat');

Test_Matrix_raw=Test_Matrix(:,2:11);
a=Test_Matrix(:,10)-0.1;
b=Test_Matrix(:,2:9);
c=Test_Matrix(:,11);
Test_Matrix_Treshold=[b,a,c];

realclass = Test_Matrix(:,1);
yfit1 = trainedClassifier.predictFcn(Test_Matrix_raw); % model 1
yfit2 = trainedClassifier.predictFcn(Test_Matrix_Treshold); % treshold model

k = find(~realclass);
k1 = realclass(k,:);
k2 = yfit1(k,:);
k3 = yfit2(k,:);

X=[k,k1,k2];
disp('Model 1');
disp('n°student Real Predicted')
disp(X)

X=[k,k1,k3];
dropouts_detected = (length(find(~k2))*100)/length(k);
fprintf('Percentage of students at risk of dropping out detected');
disp(dropouts_detected) 

disp('Treshold model');
disp('n°student Real Predicted')
disp(X)

dropouts_detected = (length(find(~k3))*100)/length(k);
fprintf('Percentage of students at risk of dropping out detected');
disp(dropouts_detected) 

