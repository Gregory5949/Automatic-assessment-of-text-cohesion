var data = []
var token = ""

jQuery(document).ready(function () {
    let on_reader_load = ( fl ) => {
        console.info( '. file reader load', fl );
        return display_file; // a function
        };

    $(document.getElementById("uploader-file")).on( "change", function (){
        value = this.files[0]
        fr = new FileReader()
        fr.readAsText(value)
        fr.onload = () => {
            let val = fr.result
            

            req = $.ajax({
                url: '/predict',
                type: 'POST',
                dataType: "json",
                contentType: "application/json",
                data:  JSON.stringify({"sentence": val, "file": true}),
                beforeSend: function () {
                $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }

                }).done(function (jsondata, textStatus, jqXHR) {

                $('#final-score').val(jsondata['response']['result'])
                }).fail(function (jsondata, textStatus, jqXHR) {
                    alert(jsondata['responseJSON'])
                });
                }
        });
    $('#btn-process').on('click', function () {
        review = $('#txt_review').val()
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",

            data: JSON.stringify({
                "sentence": review, "file": false,
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {

            $('#final-score').val(jsondata['response']['result'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON'])
        });
    })

    $('#txt_review').keypress(function (e) {
        if (e.which === 13) {
            $('#btn-process').click()
            e.preventDefault()
        }
    });
})