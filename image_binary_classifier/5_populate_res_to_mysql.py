import yaml
import MySQLdb


def create_connection_mysql(conf):
    db_name = conf["db_name"]
    HOSTNAME = conf["host"]
    USERNAME = conf["username"]
    PASSWD = conf["password"]
    try:
        conn = MySQLdb.connect(host=HOSTNAME, user=USERNAME, passwd=PASSWD , db=db_name)
        return conn
    except:
        print("connection to mysql error")
        exit()


def main():
    conf = yaml.load(open("./config.yaml"))
    conn_mysql = create_connection_mysql(conf["mysql"])
    mysql_c = conn_mysql.cursor()

    cnt = 0
    for line in open("predicts_sigmoid_no_label_imgs.txt"):
        url_md5, score = line.strip().split(",")
        score = float(score)
        if score < 0.5:
            continue
        insert_query = "UPDATE images SET binary_pred_score = %f WHERE id = '%s';" %(score, url_md5)
        mysql_c.execute(insert_query)
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
    conn_mysql.commit()
    print("update finish")


if __name__ == "__main__":
    main()
