﻿using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public class Layer
    {
        // TODO: Перенести NeuronType в класс Layer из класса Neuron
        public List<Neuron> Neurons { get; } //Коллекция нейронов слоя
        public int NeuronCount => Neurons?.Count ?? 0; //Количество нейронов в слое
        public NeuronType Type { get; }

        /// <summary>
        /// Создание слоя нейронов
        /// </summary>
        /// <param name="neurons"></param> коллекция нейронов
        /// <param name="type"></param> тип нейронов в этом слое
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            //Проверка на соответствие типу создаваемого слоя
            bool ok = false;
            for (int i = 0; i<neurons.Count;i++)
            {
                if(neurons[i].NeuronType != type)
                {
                    MessageBox.Show($"Layer contains zero neurons.\n" +
                        $"Incorrect collection '{neurons}' neuron by index {i} type in constructor parameters", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    ok = false;
                    break;
                }
                else
                {
                    ok = true;
                }
            }

            if (ok)
            {
                Neurons = neurons;
                Type = type;
            }
            else
            {
                Neurons = null;
            }
        }

        /// <summary>
        /// Получение всех выходных сигналов нейронов в слое
        /// </summary>
        /// <returns></returns>
        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach(var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
            //TODO: реализовать через yeild
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
