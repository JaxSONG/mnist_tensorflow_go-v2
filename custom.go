package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"

	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	imgPath := "./images.png"
	_, err := ioutil.ReadFile(imgPath)

	if err != nil {
		log.Fatalln(err)
	}

	root := tg.NewRoot()
	img1 := image.ReadPNG(root, imgPath, 1)
	img1 = img1.ResizeArea(image.Size{Height: 28, Width: 28})
	results := tg.Exec(root, []tf.Output{img1.Value()}, nil, &tf.SessionOptions{})
	tensor := results[0]
	fmt.Printf("DataType:%#v\n", tensor.DataType())
	fmt.Printf("shape:%#v\n", tensor.Shape())

	buf := new(bytes.Buffer)
	n, err := tensor.WriteContentsTo(buf)
	if err != nil {
		fmt.Println(err)
	}
	if n != int64(buf.Len()) {
		fmt.Printf(" WriteContentsTo said it wrote %v bytes, but wrote %v", n, buf.Len())
	}
	//    t2, err := ReadTensor(t1.DataType(), t1.Shape(), buf)
	t2, err := tf.ReadTensor(tensor.DataType(), []int64{1, 28 * 28}, buf)
	if err != nil {
		fmt.Printf("%v", err)
	}
	fmt.Printf("DataType:%#v\n", t2.DataType())
	fmt.Printf("shape:%#v\n", t2.Shape())

	model, err := tf.LoadSavedModel("mnistmodel", []string{"serve"}, nil)

	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	tensor = t2
	/*
		tensor, terr := dummyInputTensor(28 * 28) // replace this with your own data
		if terr != nil {
			fmt.Printf("Error creating input tensor: %s\n", terr.Error())
			return
		}
		fmt.Printf("dummyInputTensor DataType:%#v\n", tensor.DataType())
		fmt.Printf("dummyInputTensor shape:%#v\n", tensor.Shape())
	*/
	result, runErr := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("imageinput").Output(0): tensor,
		},
		[]tf.Output{
			model.Graph.Operation("infer").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		return
	}

	fmt.Printf("Most likely number in input is %v \n", result[0].Value())

}

func dummyInputTensor(size int) (*tf.Tensor, error) {
	rsize := make([]float32, size)
	imageData := [][]float32{rsize}
	return tf.NewTensor(imageData)
}
