# deep-photo-styletransfer
Jupyter Notebook to train photorealistic style transfer

This is an adaptation of [https://github.com/martinbenson/deep-photo-styletransfer](https://github.com/martinbenson/deep-photo-styletransfer), which is derived from [https://github.com/luanfujun/deep-photo-styletransfer](https://github.com/luanfujun/deep-photo-styletransfer)

## Steps to run style transfer

```
$ git clone https://github.com/floydhub/deep-photo-styletransfer.git deep-photo-styletransfer
$ cd deep-photo-styletransfer

# Ensure you have floyd-cli installed and you are logged into the FloydHub

$ floyd init deep-photo-styletransfer
$ floyd run --gpu --env torch --mode jupyter --data floydhub/datasets/vgg-ilsvrc-19-layers/1:models
```

This will spin up a Jupyter notebook with Torch preinstalled and the VGGNet model mounted at `/models`

### Steps to run the model

```
# Install the pre-requisites
$ !sudo apt-get install protobuf-compiler -y

# Install Lua dependencies
$ !luarocks install csvigo && luarocks install loadcaffe

# Create symlinks from the mounted data to the place the code expects it to be at
$ !cp --symbolic-link /models/* ./models

# Run the model
$ !python3 deep_photo.py -content_image examples/waterfront.png -content_seg examples/waterfront_seg.png -style_image examples/city_night.png -style_seg examples/city_night_seg.png -laplacian examples/waterfront700.csv -output_image examples/waterfront_city_night.png -image_size 700 -gpu 0
```