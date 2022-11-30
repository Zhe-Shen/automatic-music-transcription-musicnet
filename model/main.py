from preprocess import get_data
from lstm import get_text_model

def main():
    x_train, y_train = get_data('../data/data_16K/train_data', '../musicnet/train_labels')
    x_test, y_test = get_data('../data/data_16K/test_data', '../musicnet/test_labels')
    
    args = get_text_model()
    args.model.fit(
        x_train, y_train,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(x_test, y_test)
    )
    
if __name__ == '__main__':
    main()