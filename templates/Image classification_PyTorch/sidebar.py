import os

import streamlit as st

# Define possible models in a dict.
# Format of the dict:
# option 1: model -> code
# option 2 – if model has multiple variants: model -> model variant -> code
MODELS = {
    # FROM 'guochengqian/openpoints' repo...
    "Baafnet": "baafnet",
    "Ball_dgcnn": "ball_dgcnn",
    "CurveNet": "curvenet",
    "Debub_invvt": "debub_invvt",
    "Deepgcn": "deepgcn",
    "Dgcnn": "dgcnn",
    "Graphvit3d": "graphvit3d",
    "Grouppointnet": "grouppointnet",
    "Pct": "pct",
    "Pointmlp": "pointmlp",

    # DEFAULTS
    "AlexNet": "alexnet",  # single model variant
    "ResNet": {  # multiple model variants
        "ResNet 18": "resnet18",
        "ResNet 34": "resnet34",
        "ResNet 50": "resnet50",
        "ResNet 101": "resnet101",
        "ResNet 152": "resnet152",
    },
    "DenseNet": {
        "DenseNet-121": "densenet121",
        "DenseNet-161": "densenet161",
        "DenseNet-169": "densenet169",
        "DenseNet-201": "densenet201",
    },
    "VGG": {
        "VGG11": "vgg11",
        "VGG11 with batch normalization": "vgg11_bn",
        "VGG13": "vgg13",
        "VGG13 with batch normalization": "vgg13_bn",
        "VGG16": "vgg16",
        "VGG16 with batch normalization": "vgg16_bn",
        "VGG19": "vgg19",
        "VGG19 with batch normalization": "vgg19_bn",
    },
}

# 샘플링 기법 선택
SAMPLING_TECHNIQUES = {
    "FPS": "fps",
    "RS": "rs",
    "UVS": "uvs",
    "VFPS": "vfps",
    "3DEFS": "3defs",
}

# 샘플링 포인트 갯수 선택
SAMPLING_NUMBERS = {
    "1024": 1024,
    "2048": 2048,
    "4096": 4096,
    "8192": 8192,
}

# Normalize 기법 종류 설정
NORMALIZE_TECHNIQUES = {
    "Min-Max normalization": "Min-Max normalization",
    "Standardization": "Standardization",
    "L2 normalization": "L2 normalization",
}

# Define possible optimizers in a dict.
# Format: optimizer -> default learning rate
OPTIMIZERS = {
    "Adam": 0.001,
    "Adadelta": 1.0,
    "Adagrad": 0.01,
    "Adamax": 0.002,
    "RMSprop": 0.01,
    "SGD": 0.1,
}


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write("## Model")
        model = st.selectbox("Which model?", list(MODELS.keys()))

        # Show model variants if model has multiple ones.
        if isinstance(MODELS[model], dict):  # different model variants
            model_variant = st.selectbox("Which variant?", list(MODELS[model].keys()))
            inputs["model_func"] = MODELS[model][model_variant]
        else:  # only one variant
            inputs["model_func"] = MODELS[model]

        inputs["pretrained"] = st.checkbox("Use pretrained weight")
        if inputs["pretrained"]:
            # weight 파일 업로드
            uploaded_weight = st.file_uploader('Upload pretrained weight')
            if uploaded_weight is not None:
                base_dir = 'data/pretrained_weight/'
                save_uploaded_file(directory=base_dir, file=uploaded_weight)
                inputs["uploaded_pretrained_weight_dir"] = base_dir + uploaded_weight.name

        st.write("## Parameters")
        inputs["gpu"] = st.checkbox("Use GPU if available", True)
        inputs["checkpoint"] = st.checkbox("Save model checkpoint each epoch")
        if inputs["checkpoint"]:
            st.markdown(
                "<sup>Checkpoints are saved to timestamped dir in `./checkpoints`. They may consume a lot of storage!</sup>",
                unsafe_allow_html=True,
            )

        # 샘플링 기법 선택
        inputs["sampling_technique"] = st.selectbox(
            "Which sampling technique?", list(SAMPLING_TECHNIQUES.keys())
        )
        # 샘플링 포인트 갯수 선택
        inputs["sampling_number"] = st.selectbox(
            "How many points?", list(SAMPLING_NUMBERS.keys())
        )
        # Normalize 기법 선택
        inputs["normalize_technique"] = st.selectbox(
            "Which normalize technique?", list(NORMALIZE_TECHNIQUES.keys())
        )

        # Optimizer 선택
        inputs["optimizer"] = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))

        st.write("## Input data")
        inputs["data_format"] = st.selectbox(
            "Which data do you want to use?",
            ("Public dataset", "Custom dataset"),
        )
        if inputs["data_format"] == "Custom dataset":
            st.write(
                """
            Expected format: One folder per class, e.g.
            ```
            custom-data-dir
            +-- train
            |   +-- lassie.jpg
            |   +-- komissar-rex.png
            +-- test
            |   +-- garfield.png
            |   +-- smelly-cat.png
            ```
            """
            )
            # zip 파일 업로드
            uploaded_data = st.file_uploader('Upload data', type="zip")
            if uploaded_data is not None:
                base_dir = 'data/input_data/'
                save_uploaded_file(directory=base_dir, file=uploaded_data)
                inputs["uploaded_custom_zip_data_dir"] = base_dir + uploaded_data.name

        elif inputs["data_format"] == "Public dataset":
            inputs["dataset"] = st.selectbox(
                "Which one?", ("MNIST", "FashionMNIST", "CIFAR10")
            )

        st.write("## Mode")
        inputs["mode"] = st.selectbox(
            "Select mode",
            ("Train", "Test"),
        )
        if inputs["mode"] == "Train":
            validation_accuracy = st.slider("Validation accuracy", 0.0, 1.0, 0.0)
            inputs["validation_accuracy"] = validation_accuracy
            st.markdown(
                '<sup>View by running: `aim up`</sup>',
                unsafe_allow_html=True,
            )

        submit_btn = st.button("Submit")
        clear_btn = st.button("Clear All")

        if submit_btn:
            print(inputs)
            # TODO: Train/Test 실행
            # uuid 로 이름을 지정해서 ML model 에 전달
            # ex) uploaded_data.name = str(uuid.uuid4()) + ".zip"

            # TODO: 결과 출력

            # 업로드한 파일 삭제
            if "uploaded_custom_zip_data_dir" in inputs.keys():
                print(os.path.abspath(inputs["uploaded_custom_zip_data_dir"]))
                os.remove(inputs["uploaded_custom_zip_data_dir"])
            if "uploaded_pretrained_weight_dir" in inputs.keys():
                print(os.path.abspath(inputs["uploaded_pretrained_weight_dir"]))
                os.remove(inputs["uploaded_pretrained_weight_dir"])

            st.success("Done!")

        if clear_btn:
            print("Clear All")
            # TODO: Clear All

    return inputs


if __name__ == "__main__":
    show()


def save_uploaded_file(directory, file):
    # make directory if not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    absolute_path = os.path.abspath(directory)
    with open(os.path.join(absolute_path, file.name), 'wb') as f:  # 해당 경로의 폴더에서 파일의 이름으로 생성하겠다.
        f.write(file.getbuffer())  # 해당 내용은 Buffer로 작성하겠다.
        return st.success('Uploaded a file: {}'.format(file.name))
