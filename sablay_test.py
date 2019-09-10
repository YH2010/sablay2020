
# Load the model
model.load_state_dict(torch.load('output/090719230802/model_200.pth'))
model.eval()
#iter_loss = 0
#correct = 0

#for i, (inputs, labels) in enumerate(test_load):

    #inputs = Variable(inputs)
    #labels = Variable(labels)

    #if torch.cuda.is_available():
        #model = nn.DataParallel(model)
        #model.cuda()
        #inputs = inputs.cuda()
        #labels = labels.cuda()

    #optimizer.zero_grad()
    #outputs = model(inputs)
    #_, predicted = torch.max(outputs, 1)
    #loss = loss_fcn(outputs, labels)

    #iter_loss += loss.data.item()

    #correct += (predicted == labels).sum()

#test_loss = 
#sys.stdout.write("Test Loss %s, Test Accuracy %s\n" % (test_loss, test_accuracy)
#sys.stdout.flush()

file_name = sys.argv[1]
image = Image.open(file_name)
image = transform(image).float()
image = Variable(image, requires_grad=True)
image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#print(model)
#print(model(image))
outputs = model(image)
_, predicted = torch.max(outputs, 1)
print(predicted)
