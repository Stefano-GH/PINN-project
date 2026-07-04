import configparser as cp
import datetime as dt
import os

# Import configuration
conf = cp.ConfigParser()
conf.read("config.ini")

file_max_size = int( conf['log']['file_max_size'] )
log_filename = conf['log']['log_filename']


########################################
# DEFINITION LOGGER class_nameS
########################################
class Logger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


    def __init__(self, filepath):
        self.filepath = filepath

        self.log_file = os.path.sep.join([filepath, log_filename])
        self.writer = self.open_file()


    def open_file(self):

        # Check if the file path exists
        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)
        
        # Check for the size of the file
        if os.path.exists(self.log_file):
            file_size = os.path.getsize(self.log_file)
            
            if file_size >= file_max_size:
                files = [file for file in os.listdir(self.filepath) if log_filename in file]
                n_files = len(files)
                os.rename(self.log_file, f'{self.log_file}.{n_files}')
        
        file = open(self.log_file, 'a')
        return file
    

    def close_file(self):
        self.writer.close()


    def info(self, class_name, message):
        now = dt.datetime.now().strftime('%m-%d-%Y %H:%M:%S')
        self.writer.writelines('{} [INFO] {}\t{}\n'.format(now, class_name, message))
        print('[{}INFO{}] {}\t{}'.format(self.OKBLUE, self.ENDC, class_name, message))
    

    def warning(self, class_name, message):
        now = dt.datetime.now().strftime('%m-%d-%Y %H:%M:%S')
        self.writer.writelines('{} [WARNINGs] {}\t{}\n'.format(now, class_name, message))
        print('[{}WARNING{}] {}\t{}\n'.format(self.WARNING, self.ENDC, class_name, message))
    

    def error(self, class_name, message):
        now = dt.datetime.now().strftime('%m-%d-%Y %H:%M:%S')
        self.writer.writelines('{} [ERROR] {}\t{}\n'.format(now, class_name, message))
        print('[{}ERROR{}] {}\t{}\n'.format(self.FAIL, self.ENDC, class_name, message))