using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SklearnOnnxApi.Controllers
{
    [ApiController]
    [Route("/predict")]
    public class HappyClassificationController : ControllerBase
    {
        private InferenceSession _session;

        public HappyClassificationController(InferenceSession session)
        {
            _session = session;
        }

        [HttpPost]
        public ActionResult Score(HappyClassificationData data)
        {
            var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("feature_input", data.AsTensor())
            });
            Tensor<Int64> score = result.First().AsTensor<Int64>();
            var prediction = new Prediction { PredictedValue = score.First() };
            result.Dispose();
            return Ok(prediction);
        }
    }
}
