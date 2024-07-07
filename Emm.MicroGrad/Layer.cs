namespace Emm.MicroGrad;
public record Layer
{
    public Neuron[] Neurons { get; set; }
    public Layer(int inputs, int outputs)
    {
        Neurons = new Neuron[outputs];
        for (int i = 0; i < Neurons.Length; i++)
        {
            Neurons[i] = new Neuron(inputs);
        }
    }

    public Value[] Parameters()
    {
        return Neurons.SelectMany(n => n.Parameters()).ToArray();
    }

    public Value[] Apply(Value[] values)
    {
        return Neurons.Select(n => n.Apply(values)).ToArray();
    }
}
