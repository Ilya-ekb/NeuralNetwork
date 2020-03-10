using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Topology
    {
        public int InputCount { get; }//Количество входов в нейроную сеть
        public int OutputCount { get; }//Количество выходов нейронной сети
        public double LearninRate { get; }//Коэффициент скорости обучения сети
        public List<int> HiddenLayers { get; }// Количество скрытых слоев

        /// <summary>
        /// Задание топологии нейросети
        /// </summary>
        /// <param name="inputCount"></param>Количество входов в нейронную сеть
        /// <param name="outputCount"></param>Количество выходов нейронной сети
        /// <param name="layers"></param>Количество нейронов в скрытых слоях
        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            //TODO: добавить проверку вводимых данных
            InputCount = inputCount;
            OutputCount = outputCount;
            LearninRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
