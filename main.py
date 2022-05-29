from tkinter import Tk, Label, Canvas, Button, filedialog
from PIL import ImageTk, Image
import numpy as np

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Dropout
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def accuracy(y_true, y_pred):
    acc2 = accuracy_score(y_true, y_pred)
    return acc2

def load_model():
    mobile = MobileNet()

    x = mobile.layers[-6].output
    x = Dropout(0.25)(x)

    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)

    for layer in model.layers[:-23]:
        layer.trainable = False

    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
    model.load_weights('model.h5')

    return model


def browse():
    path = filedialog.askopenfilename(filetypes=[("Image File", '.png'), ("Image File", '.jpg')])
    if path not in ["", None]:
        return path


class App(Tk):
    def __init__(self):
        super().__init__()
        self.model = load_model()
        self.labels = [
            """psoriasis
            The medicines :
            *)Ciclosporin is a medicine that suppresses your immune system (immunosuppressant).
            It's usually taken daily.
            *)Acitretin is an oral retinoid that reduces skin cell production.
            *)Apremilast and dimethyl fumarate are medicines that help to reduce inflammation.
            """
            , """measles
            The treatment :
            *)WHO-recommended oral rehydration solution.
            *)two doses of vitamin A supplements, given 24 hours apart.
            """
            , """melanoma
            Medicines used include: ipilimumab ,nivolumab ,pembrolizumab ,talimogene ,laherparepvec
            """
            , """ringworm
            medicines include: clotrimazole (Canesten) ,econazole ,miconazole ,terbinafine (Lamisil)
            """
        ]

        self.title("Skin Disease Detection")

        self.label = Label(self, fg='Red', text="")
        self.label.grid(row=0, column=0, padx=(100, 100), pady=(10, 10))

        self.canvas = Canvas(self, width=200, height=200)
        self.canvas.create_image(0, 0, anchor='nw', image="", tags='image')
        self.canvas.grid(row=1, column=0, padx=(100, 100), pady=(10, 10))

        Button(self, text="Load Image", command=self.load_image).grid(row=2, column=0, padx=(50, 50), pady=(25, 10))

    def load_image(self):
        path = browse()

        if path:
            image = Image.open(path)
            image.thumbnail((200, 200))
            image = ImageTk.PhotoImage(image)
            self.canvas.itemconfig('image', image=image)
            self.canvas.img = image

            self.predict(path)

    def predict(self, path):
        image = load_img(path, target_size=(224, 224))

        image_data = img_to_array(image)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = preprocess_input(image_data)

        predict = self.model.predict(image_data)
        features = np.array(predict)
        indice_var = features.argmax(axis=-1)

        result = self.labels[max(indice_var[0][0])]
        self.label.config(text=f"The Result is: {result}")

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
