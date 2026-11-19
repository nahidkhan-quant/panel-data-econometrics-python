
View(transactions)

# Model:
model=log(price)~nocondom

# b) Pooled OLS estimates:
POLS<-plm(model,data=transactions,index=c("id","trans"),model="pooling")
summary(POLS) #note variable in the reported results


# d) Update the original model by adding attractive and age variables
model1<-log(price)~nocondom+attractive+age # model that we will be estimating
POLS<-plm(model1,data=transactions,index=c("id","trans"),model="pooling")
summary(POLS) #note variables in the reported results


#E
RE<-plm(model1,data=transactions,index=c("id","trans"),effect="individual",model = "random")
summary(RE)

#G
FE<-plm(model1,data=transactions,index=c("id","trans"),effect="individual",model="within")
summary(FE)

#I 
phtest(RE,FE)

