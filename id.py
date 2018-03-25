
keras

metrics - geen accuracy

model.fit() - arg voor hyper param optimization





denoising during training?
denoising autoencoders



minibatches nvidea

layer: average of each x,y on a batch
	create a 'plausible' reconstruction from that batch



adverserial examples

player 1 generates examples that will be misclassified by player2

autoencoder
	input-img -> encoder1 -> player2.predict() -> label /= real_label

"""
shuffle existing images
  - use as negative examples
    (probability that the shuffled picture is something is -inf)

 effect: 
 	decrease effect of background-color (e.g. lot of green)


alleen de correlatie structuur en kleur is belangrijk
	- niet kleur alleen



flip/rotate images
 - increase chance of encountering a new image from a known angle

foto's spiegelen






svm + nn

noise toevoegen als negative training data

abstracte vormen als training data?

foto's distorten/buigen/rekken om data uit te bereiden

edge-detection -> foto's croppen


advanced

generate images that activate the nn
- use these images as negative examples to increase the amount of training data




-----------------


autoencoder
	Model(encoder -> decoder -> classifier)


	Model(classifier)		with input 	x -> y

	(no labels needed)
	Model(encoder, decoder) 	with input	x -> x

Train just the decoder, feeze encoder
	Model(decoder) 	with input encode(x) -> x






Finally:
	Model(classifier) with input	decode(encode(x)) -> y


New model
	# (freeze the encoder, train a new classifier)
	# user larger hidden_layer 
	Model(classifier2) 	with input	encode(x) -> y



--- Testing (new data)


  (no labels needed)
  (more input)
  Model(encoder3, decoder3) 	with input	x3 -> x3 and x -> x


	train
	Model(encoder3,classfier3) with input 	x -> y


	predict
	Model(encoder3,classifier3) with input	x3




"""










