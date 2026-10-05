paper to check improve on medmamba when adding hsi layer

we use two dataset pad(use on original paper) and hsi(which includes rbg and hsi images)

what we did was first spand medmamba model to be able to handle hsi images regardless of the number of bands, this means that any image could be process by the model, but this bring a lot of parameters, because of this we implemented trm on it which did help reducing the number of parameter drasticly without lossing performance nor accuracy

we run experiments on one of the original dataset use by medmamba to make a comparation between both models, and agains hsi dataset to test the improvement of metrics of models

improvements, basically what gmedmamba_r is doing better than the inspiration, and also what does it brings to the fild

issues, the parts of the model that need revision or that need to bee address since they could by the reason why this is not having good result or performance
