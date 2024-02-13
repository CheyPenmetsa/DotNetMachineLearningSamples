using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace SklearnRegressionApi.Controllers
{
    [ApiController]
    [Route("/predict")]
    public class CarPriceController : ControllerBase
    {
        private InferenceSession _session;

        public CarPriceController(InferenceSession session)
        {
            _session = session;
        }

        [HttpPost]
        public ActionResult Score(CarData data)
        {
            var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Brand",  new DenseTensor<string>(new string[] {data.Brand }, new int[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Model",  new DenseTensor<string>(new string[] {data.Model }, new int[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Year",  new DenseTensor<Int64>(new Int64[] {data.Year}, new int[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Status",  new DenseTensor<string>(new string[] {data.Status }, new int[] { 1, 1 })),
                NamedOnnxValue.CreateFromTensor("Mileage",  new DenseTensor<float>(new float[] {data.Mileage }, new int[] { 1, 1 }))
            });
            var rawResult = (DisposableNamedOnnxValue)result.ToArray()[0];
            var onnxValue = rawResult.Value;
            result.Dispose();
            return Ok(onnxValue);
        }
    }
}
