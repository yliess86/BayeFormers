squad-bootstrap:
	mkdir -p dataset/squadv1
	wget  -O dataset/squadv1/train-v1.1.json https://raw.githubusercontent.com/nate-parrott/squad/master/data/train-v1.1.json
	wget  -O dataset/squadv1/dev-v1.1.json https://raw.githubusercontent.com/nate-parrott/squad/master/data/dev-v1.1.json
