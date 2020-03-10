using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }//Топология нейросети
        public List<Layer> Layers { get; }//Количество слоев сети

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        /// <summary>
        /// Вычисление выходных сигналов послойно во всех скрытых слоях и выходном слое
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            //Перебор всех слоев
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];  //Получение слоя
                var previousLayerSignals = Layers[i - 1].GetSignals(); //Получение всех выходов на предыдущем слое
                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals); //Вычисляем выходные значения на нейронах текущего слоя
                }
            }
        }

        /// <summary>
        /// Выисление выходных сигналов во всей нейросети
        /// </summary>
        /// <param name="inputSignals"></param>Входные сигналы
        /// <returns></returns>
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];//если выходное значение одно - выводим
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First(); //Сортируем значения Output выходного нейрона по убыванию и выводим самое большое
            }
        }

        /// <summary>
        /// Метод обратного распространенния ошибки 
        /// </summary>
        /// <param name="expected"></param>Ожидаемое значение
        /// <param name="inputs"></param>Входные значения
        /// <returns></returns>
        private double BackPropagationOfError(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach(var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearninRate);
            }

            for(int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for(int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for(int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearninRate);
                    }
                }
            }

            var result = difference * difference;
            return result;
        }

        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                foreach (var data in dataset)
                {
                    error += BackPropagationOfError(data.Item1, data.Item2);
                }
            }

            var result = error / epoch;
            return result;
        }

        /// <summary>
        /// Отправка входных сигнлов на входной слой нейронной сети
        /// </summary>
        /// <param name="inputSignals"></param>Входные сигналы
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            //Если количество входных сигналов и топологий сети соответствует
            if (inputSignals.Length == Topology.InputCount)
            {
                for (int i = 0; i < inputSignals.Length; i++)
                {
                    var signal = new List<double>() { inputSignals[i] };//Получаем значение входного сигнала
                    var neuron = Layers[0].Neurons[i]; //Получаем нейрон во входном слое

                    neuron.FeedForward(signal); //Вычисляем выходной сигнал нейрона
                }
            }
            else
            {
                MessageBox.Show("Number of input singals and the neural network topology do not match."                         //
                    + $"Number of input signals: {inputSignals.Length}\nNeural network topology inputs: {Topology.InputCount}",  //Обработка исключения
                    "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);                                                       //
            }
        }

        /// <summary>
        /// Созданеие выходного слоя нейросети
        /// </summary>
        private void CreateOutputLayer()
        {
            //Добавление слоя выходных нейронов в соответствии с топологией сети
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();                                  //Ссылка на предыдущий слой
            for (int i = 0; i < Topology.OutputCount; i++)                  //
            {                                                               //Создание коллекции выходных нейронов
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);//Количество входов выходных нейронов равно количеству нейронов на предыдущем слое
                outputNeurons.Add(neuron);                                  //
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);//Добавление выходного слоя в коллецию слоев нейронной сети
        }

        /// <summary>
        /// Создание скрытых слоев
        /// </summary>
        private void CreateHiddenLayers()
        {
            //Добавление слоя скрытых нейронов в соответствии с топологией сети
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)               //Внешний цикл в соответствии с количиством скрытых слоев
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();                                  //Ссылка на предыдущий слой
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)              //Внутренний цикл в соответствии с количеством нейронов в слое
                {                                                               //Создание коллекции выходных нейронов
                    var neuron = new Neuron(lastLayer.NeuronCount);                   //Количество входов скрытых нейронов равно количеству нейронов на предыдущем слое
                    hiddenNeurons.Add(neuron);                                  //
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);//Добавление скрытого слоя в коллецию слоев нейронной сети
            }
        }

        /// <summary>
        /// Создание входного слоя
        /// </summary>
        private void CreateInputLayer()
        {
            //Добавление слоя входных нейронов в соответствии с топологией сети
            var inputNeurons =new List<Neuron>();
            for(int i = 0; i < Topology.InputCount; i++)        //
            {                                                   //Создание коллекции входных нейронов
                var neuron = new Neuron(1, NeuronType.Input);   //Количество входов входного нейрона всегда равно 1
                inputNeurons.Add(neuron);                       //
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);//Добавление входного слоя в коллецию слоев нейронной сети
        }
    }
}
