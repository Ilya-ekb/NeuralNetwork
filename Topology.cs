using System.Collections.Generic;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public class Topology
    {
        public int InputCount { get; } //Количество входов в нейроную сеть
        public int OutputCount { get; } //Количество выходов нейронной сети
        public List<int> HiddenLayers { get; }// Количество скрытых слоев

        /// <summary>
        /// Задание топологии нейросети
        /// </summary>
        /// <param name="inputCount"></param>Количество входов в нейронную сеть
        /// <param name="outputCount"></param>Количество выходов нейронной сети
        /// <param name="layers"></param>Количество нейронов в скрытых слоях
        public Topology(int inputCount, int outputCount, params int[] layers)
        {
            Handler(inputCount, "Inputs", InputCount);
            Handler(outputCount, "Outputs", OutputCount);
            if (layers.Length > 0)
            {
                int[] layersChecked = default;
                for (int i = 0; i < layers.Length; i++)
                {
                    Handler(layers[i], "Count neuron on layer No.{i}", layersChecked[i]);
                }
                HiddenLayers = new List<int>();
                HiddenLayers.AddRange(layersChecked);
            }
        }
        void Handler(int value, string name, int field)
        {
            if (value > 0)
            {
                field = value;
            }
            else
            {
                MessageBox.Show($"{name} most be more than 0", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }
    }
}
