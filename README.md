# Teaching a RNN to create poems in the style of "Mihai Eminescu"

Project inspired by Martin Gorner's Tensorflow and DeepLearning presentation.

Mostly reused code from https://github.com/martin-gorner/tensorflow-rnn-shakespeare

*Training data downloaded from gutenberg.org*

### Description and how to use:

For training the model I used a VM on Google's ML platform. After launching an instance, download and install tensorflow and run:

```
python3 train.py
```

During training, a checkpoint is saved for every epoch is training. For generating some output select the desired checkpoint and run:

```
python3 run.py
```

Sample output:
```
De-acolo sa strange oceanului si cant
Si prin stele se-ntornasc in frunza-intelesata,
Caci pe-o ridica de aur si pe ceruri linistita,
Pe carari cu spatiu si stralucita -
Si intunecata si prin suflet de pamant.
	* * *
Si cuminte lungi cu steaua si cu spaima
Cand inima-i in somn a spus cu murmurul marirei,
Suntemile perii colonade, cum cu mult de mult mai multe si mult marunte,
Cand care sunt de mult istoriei se prefacute,
In vanturi ce sunt de fruntea de frunte pe-a lor umeri de lungi suri,
Si prin par de frunze printre stele si cu manile,
Umbra care suna din codri in frunza-i albastrul
Si pe cand cari si spuma si cu murmururi,
Asternandu-le-n cer albe si rugolii,
Si in cale-albastru si struguri, pe stralucite de ferestrite,
Unde albele ape,
Cu coardele lui cea carare,
Pe-a lor ceruri lungi de aur rusine,
Cu o muri ce suspina de lacrimi si-n care se prefacura.

Iar un cantec parc-al pururi se prefacu pirasul salbatica cea de braul
Cand in ochii-i le privire si pe pamant.
	* * *
```

There's a pretty small amount of training data so it does have occasional shortcomings (failed indentation and/or syntax):

```
In care stancile de femeie
Din neaua dintre soare.
						
Cand al meu suflet canturile de lacrimi si prea muritoare,
	De ce maritatea de fier,
Si-n orice lunce ceruri lungi de stele,
				Dintre stele,
				Un stalpi si mare,
				Un stele.
```
