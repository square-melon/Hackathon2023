from src.opt import Parser
from src.crawler import crawl
from src.dcgan.gan import train as dcgan_train
from src.dcgan.gan import gen as dcgan_gen
from src.dragan.gan import train as dragan_train
from src.dragan.gan import gen as dragan_gen

def main():
    parser = Parser()

    if parser.args.gen == False:
        if parser.args.no_crawl == False:
            crawl(parser.config)
        if parser.args.gan == 'dcgan':
            dcgan_train(parser.config, parser.args)
        elif parser.args.gan == 'dragan':
            dragan_train(parser.config, parser.args)
    if parser.args.gen:
        if parser.args.gan == 'dcgan':
            dcgan_gen(parser.config)
        elif parser.args.gan == 'dragan':
            dragan_gen(parser.config)

if __name__ == '__main__':
    main()