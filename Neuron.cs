using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; } //Веса входных сигналов
        public NeuronType NeuronType { get; } //Тип нейрона: input - принимает входные данные, normal - реализует скрытый слой сети, output - выдает результата работы сети
        public double Output { get; private set; }//выходное значение нейрона

        /// <summary>
        /// Создание нейрона
        /// </summary>
        /// <param name="inputCount"></param> количество входных сигналов
        /// <param name="type"></param> тип нейрна
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            //проверка входных параметров
            try
            {
                NeuronType = type;
            }
            catch (InvalidCastException e)
            {
                MessageBox.Show(e.Message, "Error input", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            try
            {
                if (inputCount >= 1)
                {
                    //Добаление весов
                    Weights = new List<double>();

                    for (int i = 0; i < inputCount; i++)
                    {
                        Weights.Add(1);
                    }
                }
                else
                {
                    MessageBox.Show("The number of inputs can't be less than 1", "Error input", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch(InvalidCastException e)
            {
                MessageBox.Show(e.Message, "Error input", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public Neuron(Neuron neuron)
        {
            Weights = neuron.Weights;
            NeuronType = neuron.NeuronType;
            Output = neuron.Output;
        }

        /// <summary>
        /// Получение результата вычилений нейрона
        /// </summary>
        /// <param name="inputs"></param> входные значения
        /// <returns></returns>
        public double FeedForward(List<double> inputs)
        {
            if (Weights.Count == inputs.Count)
            {
                var sum = 0.0;
                for (int i = 0; i < inputs.Count; i++)
                {
                    sum += inputs[i] * Weights[i];
                }

                Output = Sigmoid(sum);
                return Output;
            }
            else
            {
                MessageBox.Show("The number of weights and input signals of unequal","Error input",MessageBoxButtons.OK,MessageBoxIcon.Error);
            }

            return default;
        }

        /// <summary>
        /// Вычисление сигмоида числа
        /// </summary>
        /// <param name="x"></param> входящее число
        /// <returns></returns>
        double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public void SetWeights(params double[] weights)
        {
            //TODO: удалить после добавления возможности обучения сети
            for(int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        /// <summary>
        /// Вывод значения вычислений нейрона на экран
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return Output.ToString();
        } 
    }
}
