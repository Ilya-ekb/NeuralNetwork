using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; } //Веса входных сигналов
        public List<double> Inputs { get; }//Входы нейрона
        public NeuronType NeuronType { get; } //Тип нейрона: input - принимает входные данные, normal - реализует скрытый слой сети, output - выдает результата работы сети
        public double Output { get; private set; }//выходное значение нейрона
        public double Delta { get; private set; }//Дельта

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
                    Inputs = new List<double>();

                    InitWeghtsRandomValue(inputCount);
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

        private void InitWeghtsRandomValue(int inputCount)
        {
            var random = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(random.NextDouble());
                }
                Inputs.Add(1);
            }
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
                for(int i = 0; i < inputs.Count; i++)
                {
                    Inputs[i] = inputs[i];
                }
                var sum = 0.0;
                for (int i = 0; i < inputs.Count; i++)
                {
                    sum += inputs[i] * Weights[i];
                }
                if (NeuronType != NeuronType.Input)
                {
                    Output = Sigmoid(sum);
                }
                else
                {
                    Output = sum;
                }
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

        double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid / (1 - sigmoid);
        }

        public void Learn(double error, double learningRate)
        {
            if(NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
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
