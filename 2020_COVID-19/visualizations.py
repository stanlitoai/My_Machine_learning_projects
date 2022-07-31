import matplotlib.pyplot as plt



#Plot training and validation accuracy values
def plot_learningCurve(history, epoch):
    
    #Plot training and validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history["accuracy"])
    plt.plot(epoch_range, history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()
    
    
    
    #Plot training and validation loss values
    plt.plot(epoch_range, history.history["loss"])
    plt.plot(epoch_range, history.history["val_loss"])
    plt.title("Model loss")
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()


plot_learningCurve(history, 10)





#Plot confusion matrix


import matplotlib
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#to reset matplotli
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

#matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, 
                                show_normed=True, figsize=(6,6))
                                
plt.tight_layout()
plt.savefig("human_act.png")
plt.show()










##VISUALIZE THE TRAINING SET

plt.scatter(X_train, y_train, color="red", label="Living Area")
plt.title("House prices in king country, WA")
plt.xlabel("Area")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()



##VISUALIZE THE TEST SET

plt.scatter(X_test, y_test, color="red", label="Living Area")
plt.plot(X_train, y_pred, color="blue", label="Price")
plt.xlabel("Area")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()




##VISUALIZE THE TRAINING SET

plt.scatter(X_train, y_train, color="red", label="Living Area")
plt.title("House prices in king country, WA")
plt.plot(X_train, y_pred, color="blue", label="Price")
plt.xlabel("Area (sq-ft)")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()



##To predict something 
area = 2400



price = model.predict([[area]])

print("House of %d sq-ft costs about $%d" % (area, price))























