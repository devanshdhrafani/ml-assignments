for i=1:56
   fname=strcat(num2str(i),'.csv');
   data=csvread(fname);
   for k=2:10
      c= kmeans(data(:,1:20),k);
      eva = evalclusters(data(:,1:20),c,'Silhouette');
      evad = evalclusters(data(:,1:20),c,'DaviesBouldin');
      silv(i,k-1)=eva.CriterionValues;
      dbv(i,k-1)=evad.CriterionValues;
   end
   [M,I]=max(silv(i,1:9));
   silv(i,10)= M;
   silv(i,11)=I+1;
   [M,I]=min(dbv(i,1:9));
   dbv(i,10)= M;
   dbv(i,11)=I+1;
end