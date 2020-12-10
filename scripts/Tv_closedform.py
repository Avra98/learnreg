##Reconstruct using a TV matrix 
##
lamd=1e-1  ##lamd is the sparsity regularizer penalty.
tv=torch.zeros(m)
tv[0]=1.0
tv[1]=-1.0
TV=b_opt*src.create_circulant(tv)
xrec=src.optimize(TV,y1,lamd)

##Get the sign patterns
z= TV @ xrec
[S0,Sp,s]=src.find_sign_pattern(z, threshold=1e-6)  ##Thresholds off values less than threshold in z. 

##Observation: Even for direct closed form reconstruction, threshold and lamd is of paramount importance. 
##Threshold is set at an optimum. This code is important because it shows the sensitivity of threshold and lamd in 
##direct reconstruction. This is one of possible reasons the graident method may not be working. 

##Use the closed form expression 
xcl=src.closed_form(S0, Sp, s, TV, lamd, y1)
print(src.MSE(xcl,x1))
plt.plot(xcl)
plt.plot(xrec)
plt.plot(x1)
plt.show()
plt.imshow(TV)
plt.show()
