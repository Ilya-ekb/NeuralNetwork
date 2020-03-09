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
        /// Выисление выходных сигналов во всей нейросети
        /// </summary>
        /// <param name="inputSignals"></param>Входные сигналы
        /// <returns></returns>
        public Neuron FeedForward(List<double> inputSignals)
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
        /// Отправка входных сигнлов на входной слой нейронной сети
        /// </summary>
        /// <param name="inputSignals"></param>Входные сигналы
        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            //Если количество входных сигналов и топологий сети соответствует
            if (inputSignals.Count == Topology.InputCount)
            {
                for (int i = 0; i < inputSignals.Count; i++)
                {
                    var signal = new List<double>() { inputSignals[i] };//Получаем значение входного сигнала
                    var neuron = Layers[0].Neurons[i]; //Получаем нейрон во входном слое

                    neuron.FeedForward(signal); //Вычисляем выходной сигнал нейрона
                }
            }
            else
            {
                MessageBox.Show("Number of input singals and the neural network topology do not match."                         //
                    + $"Number of input signals: {inputSignals.Count}\nNeural network topology inputs: {Topology.InputCount}",  //Обработка исключения
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
                var neuron = new Neuron(lastLayer.Count, NeuronType.Output);//Количество входов выходных нейронов равно количеству нейронов на предыдущем слое
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
                    var neuron = new Neuron(lastLayer.Count);                   //Количество входов скрытых нейронов равно количеству нейронов на предыдущем слое
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
