# Modal FS Testing

Modal.com appears to have an extremely fast shared file system, this reposity sets up a small test to see how performant it is.

Ideally you would want to load a ML model before any requests an cache it between requests in a serverless environment. However,
if you cannot know what model you need until runtime then it is important to be able to load it very quickly.

In this test I train an AutoEncoder on MNIST which checkpoints at about ~1.2mb (a tiny model)
and load different saves of the model based on a request parameter. After a cold start, loading the model takes
anywhere from 8-25ms for a 1mb file. I need to try different model sizes to see how this scales, but this is pretty
fast for a distributed file system in a serverless environment.
