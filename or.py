from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import numpy as np
import pandas as pd

def main(data, eta, epochs, filename, plot_name):

    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plot_name, model)

if __name__=='__main__': #<< execution will start from here

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    main(data=OR, eta = ETA, epochs=EPOCHS, filename="or.model", plot_name="or.png")
