import torch
import prototorch as pt
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class CBC(torch.nn.Module):

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.components_layer = pt.components.ReasoningComponents(
            distribution=[1, 1],
            components_initializer=pt.initializers.SSCI(data, noise=0.1),
            reasonings_initializer=pt.initializers.PPRI(components_first=True),
        )

    def forward(self, x):
        components, reasonings = self.components_layer()
        sims = pt.similarities.euclidean_similarity(x, components)
        probs = pt.competitions.cbcc(sims, reasonings)
        return probs


class VisCBC2D():

    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train = pt.utils.parse_data_arg(data)
        self.title = "Components Visualization"
        self.fig = plt.figure(self.title)
        self.border = 0.1
        self.resolution = 100
        self.cmap = "viridis"

    def on_train_epoch_end(self):
        x_train, y_train = self.x_train, self.y_train
        _components = self.model.components_layer._components.detach()
        ax = self.fig.gca()
        ax.cla()
        ax.set_title(self.title)
        ax.axis("off")
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap=self.cmap,
            edgecolor="k",
            marker="o",
            s=30,
        )
        ax.scatter(
            _components[:, 0],
            _components[:, 1],
            c="w",
            cmap=self.cmap,
            edgecolor="k",
            marker="D",
            s=50,
        )
        x = torch.vstack((x_train, _components))
        mesh_input, xx, yy = pt.utils.mesh2d(x, self.border, self.resolution)
        with torch.no_grad():
            y_pred = self.model(
                torch.Tensor(mesh_input).type_as(_components)).argmax(1)
        y_pred = y_pred.cpu().reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=self.cmap, alpha=0.35)
        plt.pause(0.2)


if __name__ == "__main__":

    scaler = StandardScaler()
    df = pd.read_csv('BankNote_Authentication.csv').to_numpy()

    X = df[:, 0: 4]
    y = df[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, y_train.shape)

    # Dataset
    train_ds = pt.datasets.NumpyDataset(X_train, y_train)
    test_ds = pt.datasets.NumpyDataset(X_test, y_test)

    print(X_train.shape)

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    # create an object of the model class
    model = CBC(train_ds)

    # choose initializers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = pt.losses.MarginLoss(margin=0.1)
    # vis = VisCBC2D(model, train_ds)

    # Train the model for n epochs
    final_losses = []
    for epoch in range(200):
        correct = 0.0
        for x, y in train_loader:
            y_oh = torch.eye(2, 2)[y]
            y_pred = model(x)
            loss = criterion(y_pred, y_oh).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (y_pred.argmax(1) == y).float().sum(0)

        acc = 100 * correct / len(train_ds)
        final_losses.append(loss)

        # print the train accuracy and loss after some epochs
        if epoch % 5 == 1:
            print(f"Epoch: {epoch} Accuracy: {acc:05.02f}% loss:{loss} ")
        # vis.on_train_epoch_end()

    # Visualize training performance
    plt.plot(range(200), final_losses)
    plt.xlabel('Train Loss')
    plt.ylabel('Number of epochs')
    plt.show()

    # Make predictions
    _components = model.components_layer._components.detach()
    y_pred = model(torch.Tensor(np.array(X_test)).type_as(_components)).argmax(1).detach().numpy()
    print(model(torch.Tensor(np.array(X_test)).type_as(_components)).argmax(1).detach().numpy())

    # Evaluate train model
    print(accuracy_score(y_test, y_pred))

    # Get components
    print(model.components_layer.components)

    # save model
    torch.save(model, 'cbc.pt')

    # load model
    torch.load('cbc.pt')
