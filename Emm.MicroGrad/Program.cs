// ReSharper disable InconsistentNaming

namespace Emm.MicroGrad;

internal class Program
{
    static Value[] ToValue(float[] values)
    {
        var result = new Value[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = new Value(values[i]);
        }

        return result;
    }
    
    static void Main(string[] args)
    {
        var mlp = new Mlp(3, [4, 4, 1]);

        Console.WriteLine(mlp.Parameters().Length);

        Value[][] xs = [
            ToValue([2.0f, 3.0f, -1.0f]),
            ToValue([3.0f, -1.0f, 0.5f]),
            ToValue([0.5f, 1.0f, 1.0f]),
            ToValue([1.0f, 1.0f, -1.0f])];
        var ys = ToValue([1.0f, -1.0f, -1.0f, 1.0f]);


        int passes = 200;
        float learningRate = -0.01f;
        Value[]? ypred = null;
        for (int k = 0; k < passes; k++)
        {
            //create prediction with new weights
            ypred = xs.Select(x => mlp.Apply(x)[0]).ToArray();

            //calculate loss function
            var loss = new Value(0.0f);
            foreach (var zip in ys.Zip(ypred))
            {
                loss = loss + (zip.Second - zip.First).Pow(2.0f);
            }

            //Zero grad
            foreach (var parameter in mlp.Parameters()) parameter.Grad = 0.0f;
            //backward pass
            loss.Backward();
            
            //Adjust weights and biases
            foreach (var parameter in mlp.Parameters())
            {
                parameter.Data += learningRate
                                  * parameter.Grad;
            }
            
            Console.WriteLine($"{k}\tloss: {loss.Data}");
        }

        if (ypred is null) return;
        for (int i = 0; i < ypred.Length; i++)
        {
            Console.WriteLine($"[{i}] {ypred[i].Data}");
        }
        
        
 

         

    }
}
